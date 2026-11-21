"""OpenAI-compatible HTTP server for the engine (one request at a time; requests queue on a lock).

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
MARKERS = ("<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>")


class Splitter:
    """Incrementally splits decoded model text into (reasoning, content) deltas."""

    def __init__(self):
        self.buf = ""
        self.mode = "content"

    def feed(self, text: str, final: bool = False) -> tuple[str, str]:
        self.buf += text
        r, c = [], []
        while self.buf:
            if self.buf.startswith(THINK):
                self.mode, self.buf = "reasoning", self.buf[len(THINK):]
                continue
            if self.buf.startswith(TEXT):
                self.mode, self.buf = "content", self.buf[len(TEXT):]
                continue
            hit = next((m for m in MARKERS if self.buf.startswith(m)), None)
            if hit:
                self.buf = self.buf[len(hit):]
                continue
            i = self.buf.find("<|")
            if i == -1:
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
            (r if self.mode == "reasoning" else c).append(chunk)
        return "".join(r), "".join(c)


class Server:
    def __init__(self, engine, tokenizer, model_id: str):
        self.engine, self.tok, self.model_id = engine, tokenizer, model_id
        self.lock = threading.Lock()
        self.stop_ids = {engine.cfg.eos_token_id}
        for t in ("<|content_model_end_sampling|>",):
            tid = tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)

    # --------------------------------------------------------------- requests
    def chat_ids(self, body: dict) -> list[int]:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if "reasoning_effort" in body:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        enc = self.tok.apply_chat_template(body["messages"], tools=body.get("tools"), tokenize=True,
                                           add_generation_prompt=True, return_dict=True, **kwargs)
        return [int(i) for i in enc["input_ids"]]

    @staticmethod
    def sampling(body: dict):
        from .engine import Sampling

        return Sampling(temperature=float(body.get("temperature", 0.0) or 0.0), top_p=float(body.get("top_p", 1.0)),
                        top_k=int(body.get("top_k", 0) or 0), seed=body.get("seed"))

    def run(self, ids, body, on_token):
        """Generate under the lock, calling on_token(token_id) per token. Returns (tokens, stats)."""
        from .engine import GenStats

        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        st = GenStats()
        out = []
        with self.lock:
            for t in self.engine.stream(ids, max_new, self.stop_ids, self.sampling(body), stats=st):
                out.append(t)
                on_token(t)
        return out, st

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

    async def generate(ids, body, chat: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        done = object()
        tok = server.tok

        def worker():
            try:
                res = server.run(ids, body, lambda t: loop.call_soon_threadsafe(q.put_nowait, t))
                loop.call_soon_threadsafe(q.put_nowait, (done, res))
            except Exception as e:  # surfaced to the client
                loop.call_soon_threadsafe(q.put_nowait, (done, e))

        threading.Thread(target=worker, daemon=True).start()
        split = Splitter()
        toks: list[int] = []
        prev_text = ""
        while True:
            item = await q.get()
            if isinstance(item, tuple) and item and item[0] is done:
                res = item[1]
                if isinstance(res, Exception):
                    raise res
                out, st = res
                r, c = split.feed("", final=True)
                yield ("delta", r, c, None)
                yield ("end", out, st, rid, created)
                return
            toks.append(item)
            text = tok.decode(toks, skip_special_tokens=False)
            delta, prev_text = text[len(prev_text):], text
            if delta.endswith("�"):  # incomplete UTF-8 sequence: wait for the next token
                prev_text = prev_text[:-1]
                delta = delta[:-1]
            r, c = split.feed(delta) if chat else ("", delta)
            if r or c:
                yield ("delta", r, c, None)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        ids = server.chat_ids(body)
        return await respond(ids, body, chat=True)

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body["prompt"]
        ids = prompt if isinstance(prompt, list) else server.tok.encode(prompt, add_special_tokens=False)
        return await respond(ids, body, chat=False)

    async def respond(ids, body, chat: bool):
        stream = bool(body.get("stream"))
        obj = "chat.completion" if chat else "text_completion"
        if stream:
            async def sse():
                async for ev in generate(ids, body, chat):
                    if ev[0] == "delta":
                        _, r, c, _ = ev
                        if not (r or c):
                            continue
                        if chat:
                            delta = {}
                            if r:
                                delta["reasoning_content"] = r
                            if c:
                                delta["content"] = c
                            choice = {"index": 0, "delta": delta, "finish_reason": None}
                        else:
                            choice = {"index": 0, "text": c, "finish_reason": None}
                        yield f"data: {json.dumps({'object': obj + '.chunk', 'choices': [choice], 'model': server.model_id})}\n\n"
                    else:
                        _, out, st, rid, created = ev
                        fin = server.finish_reason(out, body)
                        choice = ({"index": 0, "delta": {}, "finish_reason": fin} if chat
                                  else {"index": 0, "text": "", "finish_reason": fin})
                        yield f"data: {json.dumps({'id': rid, 'object': obj + '.chunk', 'created': created, 'model': server.model_id, 'choices': [choice], 'usage': server.usage(ids, out, st)})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        reasoning, content = [], []
        final = None
        async for ev in generate(ids, body, chat):
            if ev[0] == "delta":
                reasoning.append(ev[1])
                content.append(ev[2])
            else:
                final = ev
        _, out, st, rid, created = final
        fin = server.finish_reason(out, body)
        if chat:
            msg = {"role": "assistant", "content": "".join(content)}
            if any(reasoning):
                msg["reasoning_content"] = "".join(reasoning)
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
    ap.add_argument("--max-ctx", type=int, default=65536)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--dense-fp8", action="store_true")
    ap.add_argument("--model-id", default="inkling-2-mini")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    import uvicorn

    from .engine import Engine

    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(a.model, max_ctx=a.max_ctx, spec=not a.no_spec, dense_fp8=a.dense_fp8)
    uvicorn.run(build_app(Server(eng, tok, a.model_id)), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
