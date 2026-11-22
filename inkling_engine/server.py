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


def generation_defaults(model_dir: str) -> dict:
    """Sampling defaults from the checkpoint's generation_config.json (Inkling-2-Mini: do_sample, temperature 1.0),
    used for requests that do not set them, as SGLang / transformers do."""
    import os

    try:
        with open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8") as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    if not g.get("do_sample", False):
        return {"temperature": 0.0}
    return {k: g[k] for k in ("temperature", "top_p", "top_k") if k in g}


class Server:
    def __init__(self, engine, tokenizer, model_id: str, defaults: dict | None = None):
        """defaults: sampling parameters for requests that omit them (default: greedy)."""
        from .scheduler import Scheduler

        self.engine, self.tok, self.model_id = engine, tokenizer, model_id
        self.defaults = dict(defaults or {})
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

    def sampling(self, body: dict):
        from .engine import Sampling

        d = self.defaults
        pick = lambda k, dv: body[k] if body.get(k) is not None else d.get(k, dv)  # noqa: E731
        return Sampling(temperature=float(pick("temperature", 0.0) or 0.0), top_p=float(pick("top_p", 1.0)),
                        top_k=int(pick("top_k", 0) or 0), seed=body.get("seed"),
                        presence_penalty=float(body.get("presence_penalty") or 0.0),
                        frequency_penalty=float(body.get("frequency_penalty") or 0.0),
                        repetition_penalty=float(body.get("repetition_penalty") or 1.0))

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


def stop_strings(body: dict) -> list[str]:
    """The request's `stop` sequences (a string or a list of up to 4, OpenAI style); generation ends before the
    first occurrence in the output text, which is not returned."""
    stop = body.get("stop")
    if stop is None:
        return []
    stop = [stop] if isinstance(stop, str) else list(stop)
    return [x for x in stop if isinstance(x, str) and x]


def requested_logprobs(body: dict, chat: bool) -> int | None:
    """Alternatives per token to report, or None: chat uses logprobs (bool) + top_logprobs, completions an int."""
    if chat:
        return int(body.get("top_logprobs") or 0) if body.get("logprobs") else None
    lp = body.get("logprobs")
    return None if lp is None or lp is False else int(lp)


def logprobs_obj(tok, entries, chat: bool):
    """OpenAI logprobs object for [(token id, logprob, [(alt id, logprob)])]."""
    text = lambda i: tok.decode([i], skip_special_tokens=False)  # noqa: E731
    if chat:
        def one(i, lp):
            t = text(i)
            return {"token": t, "logprob": lp, "bytes": list(t.encode("utf-8"))}
        return {"content": [{**one(i, lp), "top_logprobs": [one(a, alp) for a, alp in top]}
                            for i, lp, top in entries]}
    return {"tokens": [text(i) for i, _, _ in entries], "token_logprobs": [lp for _, lp, _ in entries],
            "top_logprobs": [{text(a): alp for a, alp in top} for _, _, top in entries]}


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
        n_lp = requested_logprobs(body, chat)
        req = server.sched.submit(ids, max_new, server.stop_ids, server.sampling(body), on_token, image_feats,
                                  logprobs=n_lp)
        lp_sent = 0

        def new_lps():  # logprob entries of the tokens consumed since the last delta
            nonlocal lp_sent
            if n_lp is None:
                return None
            n = min(len(toks), len(req.lp))
            out = [(toks[j], *req.lp[j]) for j in range(lp_sent, n)]
            lp_sent = n
            return out
        split = Splitter()
        stops = stop_strings(body)
        hold = max((len(x) for x in stops), default=1) - 1  # chars kept back while a stop string may be forming
        toks: list[int] = []
        emitted = 0  # chars of the decoded text already passed on
        stopped = False
        text = ""

        def feed(delta, final=False):
            return split.feed(delta, final=final) if chat else ("", delta, [])

        try:
            while True:
                item = await q.get()
                if isinstance(item, tuple) and item and item[0] is done:
                    if req.error is not None:
                        raise req.error
                    tail = "" if stopped else text[emitted:]
                    r, c, calls = feed(tail, final=True)
                    yield ("delta", r, c, calls, new_lps())
                    yield ("end", list(out_tokens), req.stats, rid, created, stopped)
                    return
                if stopped:
                    continue  # tokens generated before the cancel reached the scheduler
                toks.append(item)
                text = tok.decode(toks, skip_special_tokens=False)
                end = len(text) - 1 if text.endswith("�") else len(text)  # incomplete UTF-8: wait
                if stops:
                    lo = max(0, emitted - hold)
                    hits = [i for i in (text.find(x, lo) for x in stops) if i >= 0]
                    if hits:
                        end, stopped = min(hits), True
                        req.cancel.set()
                    else:
                        end = max(emitted, end - hold)
                delta, emitted = text[emitted:end], max(emitted, end)
                r, c, calls = feed(delta)
                if r or c or calls:
                    yield ("delta", r, c, calls, new_lps())
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
        limit = server.engine.max_ctx - 16  # room for one verify step
        if len(ids) > limit:
            return JSONResponse({"error": {"message": f"prompt has {len(ids)} tokens; this server's context is "
                                                      f"{server.engine.max_ctx} tokens (--max-ctx)",
                                           "type": "invalid_request_error", "code": "context_length_exceeded"}},
                                status_code=400)
        stream = bool(body.get("stream"))
        obj = "chat.completion" if chat else "text_completion"
        if stream:
            n_calls = [0]

            async def sse():
                async for ev in generate(ids, body, chat, image_feats):
                    if ev[0] == "delta":
                        _, r, c, calls, lps = ev
                        if not (r or c or calls or lps):
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
                        if lps is not None:
                            choice["logprobs"] = logprobs_obj(server.tok, lps, chat)
                        yield f"data: {json.dumps({'object': obj + '.chunk', 'choices': [choice], 'model': server.model_id})}\n\n"
                    else:
                        _, out, st, rid, created, stopped = ev
                        fin = "tool_calls" if n_calls[0] else ("stop" if stopped else server.finish_reason(out, body))
                        choice = ({"index": 0, "delta": {}, "finish_reason": fin} if chat
                                  else {"index": 0, "text": "", "finish_reason": fin})
                        yield f"data: {json.dumps({'id': rid, 'object': obj + '.chunk', 'created': created, 'model': server.model_id, 'choices': [choice], 'usage': server.usage(ids, out, st)})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        reasoning, content, calls, lps = [], [], [], None
        final = None
        async for ev in generate(ids, body, chat, image_feats):
            if ev[0] == "delta":
                reasoning.append(ev[1])
                content.append(ev[2])
                calls += ev[3] or []
                if ev[4] is not None:
                    lps = (lps or []) + ev[4]
            else:
                final = ev
        _, out, st, rid, created, stopped = final
        fin = "tool_calls" if calls else ("stop" if stopped else server.finish_reason(out, body))
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
        if lps is not None:
            choice["logprobs"] = logprobs_obj(server.tok, lps, chat)
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
    ap.add_argument("--fp4-prefill", action="store_true",
                    help="NVFP4 activations + FP4 tensor cores for prefill expert GEMMs (Triton >= 3.6)")
    ap.add_argument("--no-vision", action="store_true", help="text only (skips loading the 0.6 GB vision tower)")
    ap.add_argument("--model-id", default="inkling-2-mini")
    ap.add_argument("--default-temperature", type=float, default=None,
                    help="temperature for requests without one (default: the checkpoint's generation_config, 1.0)")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import uvicorn

    from .engine import Engine

    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(a.model, max_ctx=a.max_ctx, spec=not a.no_spec, dense_fp8=a.dense_fp8,
                 kv_fp8=a.kv_fp8, slots=a.slots, vision=not a.no_vision,
                 fp4_prefill=a.fp4_prefill or None)
    defaults = generation_defaults(a.model)
    if a.default_temperature is not None:
        defaults["temperature"] = a.default_temperature
    uvicorn.run(build_app(Server(eng, tok, a.model_id, defaults)), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
