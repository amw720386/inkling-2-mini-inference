"""OpenAI-compatible HTTP server for the engine (continuous batching over --slots concurrent requests).

  python -m inkling_engine.server --model /path/to/Inkling-2-Mini --port 8000 [--no-spec] [--dense-fp8]

Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/completions (both with "stream": true).
Chat output is split into `reasoning_content` (the model's thinking section) and `content`.
`reasoning_effort` ("none".."max" or a number) sets the template's thinking effort.
"""


import argparse
import asyncio
import json
import threading
import time
import uuid


THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
HEADER = "<|message_model|>"
END = "<|end_message|>"
DROP = ("<|content_model_end_sampling|>",)


class Splitter:
    """Incrementally splits decoded model text into reasoning, content and tool calls.

    A model message is `<|message_model|>[NAME]<|content_X|>body<|end_message|>`; X = thinking -> reasoning,
    text -> content, invoke_tool_json -> a tool call whose body is {"name": ..., "args": {...}}.
    feed() returns (reasoning delta, content delta, [completed tool calls as (name, args dict)]).
    """

    def __init__(self):
        self.buf = ""
        self.mode = "content"  # content | reasoning | header | tool
        self.header = ""
        self.tool = ""

    def feed(self, text: str, final: bool = False):
        self.buf += text
        r, c, calls = [], [], []
        while self.buf:
            if self.buf.startswith(HEADER):
                self.mode, self.header, self.buf = "header", "", self.buf[len(HEADER):]
                continue
            if self.buf.startswith(THINK):
                self.mode, self.buf = "reasoning", self.buf[len(THINK):]
                continue
            if self.buf.startswith(TEXT):
                self.mode, self.buf = "content", self.buf[len(TEXT):]
                continue
            if self.buf.startswith(TOOL_JSON):
                self.mode, self.tool, self.buf = "tool", "", self.buf[len(TOOL_JSON):]
                continue
            if self.buf.startswith(END):
                if self.mode == "tool":
                    calls.append(self._tool_call())
                self.mode, self.buf = "content", self.buf[len(END):]
                continue
            hit = next((m for m in DROP if self.buf.startswith(m)), None)
            if hit:
                self.buf = self.buf[len(hit):]
                continue
            i = self.buf.find("<|")
            if i == -1:
                if self.buf.endswith("<") and not final:  # maybe the start of a special token
                    if len(self.buf) == 1:
                        break
                    chunk, self.buf = self.buf[:-1], "<"
                else:
                    chunk, self.buf = self.buf, ""
            elif i > 0:
                chunk, self.buf = self.buf[:i], self.buf[i:]
            else:
                j = self.buf.find("|>")
                if j == -1 and not final:
                    break  # partial special token: wait for more text
                end = len(self.buf) if j == -1 else j + 2
                self.buf = self.buf[end:]  # drop unknown special tokens
                continue
            if self.mode == "header":
                self.header += chunk
            elif self.mode == "tool":
                self.tool += chunk
            else:
                (r if self.mode == "reasoning" else c).append(chunk)
        if final and self.mode == "tool" and self.tool:
            calls.append(self._tool_call())
            self.mode = "content"
        return "".join(r), "".join(c), calls

    def _tool_call(self):
        try:
            obj = json.loads(self.tool)
            name, args = obj.get("name", self.header.strip()), obj.get("args", {})
        except (json.JSONDecodeError, AttributeError):
            name, args = self.header.strip(), {"_raw": self.tool}
        self.tool = ""
        return name, args


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI clients send tool-call arguments as JSON strings; the template wants parsed objects."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        fn["arguments"] = {"_raw": fn["arguments"]}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


IMAGE_TYPES = ("image", "image_url", "input_image")


def has_images(messages) -> bool:
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in IMAGE_TYPES
                                                          for p in m["content"]) for m in messages)


def _load_image(src: str):
    import base64
    import io

    from PIL import Image

    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    else:
        import urllib.request

        with urllib.request.urlopen(src, timeout=30) as f:
            data = f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_images(messages):
    """OpenAI image parts (image_url with data: or http(s) URL) -> {"type": "image", "image": PIL.Image}."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") in IMAGE_TYPES:
                    src = p.get("image_url") or p.get("image") or p.get("url")
                    if isinstance(src, dict):
                        src = src.get("url")
                    parts.append({"type": "image", "image": _load_image(src) if isinstance(src, str) else src})
                else:
                    parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out


def tool_call_obj(name, args, index):
    return {"index": index, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class Server:
    def __init__(self, engine, tokenizer, model_id: str):
        from .scheduler import Scheduler

        self.engine, self.tok, self.model_id = engine, tokenizer, model_id
        self.sched = Scheduler(engine)  # continuous batching over engine.slots
        self.stop_ids = {engine.cfg.eos_token_id}
        for t in ("<|content_model_end_sampling|>",):
            tid = tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)

    # --------------------------------------------------------------- requests
    def chat_ids(self, body: dict):
        """(input ids, image features or None) of a chat request."""
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if "reasoning_effort" in body:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        messages = normalize_messages(body["messages"])
        if has_images(messages):
            if self.engine.vision is None:
                raise ValueError("image input needs the server started with vision enabled")
            return self.engine.vision.chat_inputs(load_images(messages), tools=body.get("tools"), **kwargs)
        enc = self.tok.apply_chat_template(messages, tools=body.get("tools"), tokenize=True,
                                           add_generation_prompt=True, return_dict=True, **kwargs)
        return [int(i) for i in enc["input_ids"]], None

    @staticmethod
    def sampling(body: dict):
        from .engine import Sampling

        return Sampling(temperature=float(body.get("temperature", 0.0) or 0.0), top_p=float(body.get("top_p", 1.0)),
                        top_k=int(body.get("top_k", 0) or 0), seed=body.get("seed"))

    def run(self, ids, body, on_token):
        """Generate under the lock, calling on_token(token_id) per token. Returns (tokens, stats)."""
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        out = []

        def cb(t):
            if t is not None:
                out.append(t)
                on_token(t)

        req = self.sched.submit(ids, max_new, self.stop_ids, self.sampling(body), cb)
        req.done.wait()
        if req.error is not None:
            raise req.error
        return out, req.stats

    def usage(self, ids, out, st):
        return {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out),
                "prompt_tokens_details": {"cached_tokens": st.reused_tokens}}

    def finish_reason(self, out, body):
        return "stop" if out and out[-1] in self.stop_ids else "length"


def build_app(server: Server):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": server.model_id, "object": "model", "owned_by": "local"}]}

    @app.get("/health")
    async def health():
        return {"ok": True}

    async def generate(ids, body, chat: bool, image_feats=None):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        done = object()
        tok = server.tok

        out_tokens: list[int] = []

        def on_token(t):  # scheduler thread -> event loop
            if t is None:
                loop.call_soon_threadsafe(q.put_nowait, (done, None))
            else:
                out_tokens.append(t)
                loop.call_soon_threadsafe(q.put_nowait, t)

        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        req = server.sched.submit(ids, max_new, server.stop_ids, server.sampling(body), on_token, image_feats)
        split = Splitter()
        toks: list[int] = []
        prev_text = ""
        try:
            while True:
                item = await q.get()
                if isinstance(item, tuple) and item and item[0] is done:
                    if req.error is not None:
                        raise req.error
                    r, c, calls = split.feed("", final=True)
                    yield ("delta", r, c, calls)
                    yield ("end", list(out_tokens), req.stats, rid, created)
                    return
                toks.append(item)
                text = tok.decode(toks, skip_special_tokens=False)
                delta, prev_text = text[len(prev_text):], text
                if delta.endswith("�"):  # incomplete UTF-8 sequence: wait for the next token
                    prev_text = prev_text[:-1]
                    delta = delta[:-1]
                r, c, calls = split.feed(delta) if chat else ("", delta, [])
                if r or c or calls:
                    yield ("delta", r, c, calls)
        finally:
            if not req.done.is_set():  # the client went away (generator closed): stop generating
                req.cancel.set()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        try:
            ids, feats = server.chat_ids(body)
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=400)
        return await respond(ids, body, chat=True, image_feats=feats)

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body["prompt"]
        ids = prompt if isinstance(prompt, list) else server.tok.encode(prompt, add_special_tokens=False)
        return await respond(ids, body, chat=False)

    async def respond(ids, body, chat: bool, image_feats=None):
        stream = bool(body.get("stream"))
        obj = "chat.completion" if chat else "text_completion"
        if stream:
            n_calls = [0]

            async def sse():
                async for ev in generate(ids, body, chat, image_feats):
                    if ev[0] == "delta":
                        _, r, c, calls = ev
                        if not (r or c or calls):
                            continue
                        if chat:
                            delta = {}
                            if r:
                                delta["reasoning_content"] = r
                            if c:
                                delta["content"] = c
                            if calls:
                                delta["tool_calls"] = [tool_call_obj(n, a, n_calls[0] + i) for i, (n, a) in enumerate(calls)]
                                n_calls[0] += len(calls)
                            choice = {"index": 0, "delta": delta, "finish_reason": None}
                        else:
                            choice = {"index": 0, "text": c, "finish_reason": None}
                        yield f"data: {json.dumps({'object': obj + '.chunk', 'choices': [choice], 'model': server.model_id})}\n\n"
                    else:
                        _, out, st, rid, created = ev
                        fin = "tool_calls" if n_calls[0] else server.finish_reason(out, body)
                        choice = ({"index": 0, "delta": {}, "finish_reason": fin} if chat
                                  else {"index": 0, "text": "", "finish_reason": fin})
                        yield f"data: {json.dumps({'id': rid, 'object': obj + '.chunk', 'created': created, 'model': server.model_id, 'choices': [choice], 'usage': server.usage(ids, out, st)})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        reasoning, content, calls = [], [], []
        final = None
        async for ev in generate(ids, body, chat, image_feats):
            if ev[0] == "delta":
                reasoning.append(ev[1])
                content.append(ev[2])
                calls += ev[3] or []
            else:
                final = ev
        _, out, st, rid, created = final
        fin = "tool_calls" if calls else server.finish_reason(out, body)
        if chat:
            text = "".join(content)
            msg = {"role": "assistant", "content": text if (text or not calls) else None}
            if any(reasoning):
                msg["reasoning_content"] = "".join(reasoning)
            if calls:
                msg["tool_calls"] = [tool_call_obj(n, a, i) for i, (n, a) in enumerate(calls)]
            choice = {"index": 0, "message": msg, "finish_reason": fin}
        else:
            choice = {"index": 0, "text": "".join(content), "finish_reason": fin}
        return JSONResponse({"id": rid, "object": obj, "created": created, "model": server.model_id,
                             "choices": [choice], "usage": server.usage(ids, out, st),
                             "timings": {"prefill_s": st.prefill_s, "decode_s": st.decode_s,
                                         "decode_tok_s": st.decode_tok_s, "spec_rounds": st.rounds,
                                         "spec_accepted": st.accepted}})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--slots", type=int, default=4, help="concurrent requests (batched decode)")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--dense-fp8", action="store_true")
    ap.add_argument("--kv-fp8", action="store_true", help="FP8 cache for the full-attention layers (long contexts)")
    ap.add_argument("--no-vision", action="store_true", help="text only (skips loading the 0.6 GB vision tower)")
    ap.add_argument("--model-id", default="inkling-2-mini")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import uvicorn

    from .engine import Engine

    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(a.model, max_ctx=a.max_ctx, spec=not a.no_spec, dense_fp8=a.dense_fp8,
                 kv_fp8=a.kv_fp8, slots=a.slots, vision=not a.no_vision)
    uvicorn.run(build_app(Server(eng, tok, a.model_id)), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
