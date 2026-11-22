"""Structured output (`response_format`) with xgrammar.

Inkling writes `<|message_model|>...<|content_thinking|>reasoning<|end_message|>` before its answer, so the grammar
cannot apply from the first token. A JsonConstraint lets the model generate freely until it opens a text content
section (`<|content_text|>`), then masks every token to the JSON grammar / schema until the grammar accepts its stop
token, `<|end_message|>`, and then forces the end-of-sampling token. Constrained requests decode without speculation.
"""

from __future__ import annotations

import json
import threading

import torch

TEXT = "<|content_text|>"
END = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"


class GrammarBackend:
    """Tokenizer info and a compiler (built on first use: the vocabulary scan takes a few seconds), with a cache of
    compiled grammars per schema."""

    def __init__(self, tokenizer, vocab_size: int):
        self.tok, self.vocab_size = tokenizer, vocab_size
        self.ids = {name: tokenizer.convert_tokens_to_ids(name) for name in (TEXT, END, END_SAMPLING)}
        self._compiler = None
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _get_compiler(self):
        import xgrammar as xgr

        if self._compiler is None:
            info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.vocab_size,
                                                      stop_token_ids=[self.ids[END]])
            self._compiler = xgr.GrammarCompiler(info)
        return self._compiler

    def compile(self, response_format: dict):
        """Compiled grammar for an OpenAI response_format, or None (type "text")."""
        kind = (response_format or {}).get("type", "text")
        if kind == "text":
            return None
        if kind == "json_object":
            key = "json"
        elif kind == "json_schema":
            spec = response_format.get("json_schema") or {}
            schema = spec.get("schema", spec)
            key = json.dumps(schema, sort_keys=True)
        else:
            raise ValueError(f"unsupported response_format type {kind!r}")
        with self._lock:
            if key not in self._cache:
                c = self._get_compiler()
                self._cache[key] = (c.compile_builtin_json_grammar() if key == "json"
                                    else c.compile_json_schema(key))
            return self._cache[key]

    def constraint(self, compiled, from_start: bool = False) -> "JsonConstraint":
        """from_start: constrain from the first token (raw completions, no message structure)."""
        return JsonConstraint(compiled, self.ids, self.vocab_size, from_start)


class JsonConstraint:
    """Per-request state: free -> grammar (after `<|content_text|>`) -> forced end-of-sampling."""

    def __init__(self, compiled, ids: dict, vocab_size: int, from_start: bool = False):
        import xgrammar as xgr

        self.xgr = xgr
        self.matcher = xgr.GrammarMatcher(compiled)
        self.ids = ids
        self.state = "grammar" if from_start else "free"
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def pick(self, logits: torch.Tensor, sample) -> int:
        """Next token: sample(row) on the (masked) logits row, or a forced token."""
        if self.state == "done":
            return self.ids[END_SAMPLING]
        if self.state == "free":
            t = sample(logits)
            if t == self.ids[TEXT]:
                self.state = "grammar"
            return t
        row = logits.reshape(1, -1).float().clone()
        self.matcher.fill_next_token_bitmask(self.bitmask)
        self.xgr.apply_token_bitmask_inplace(row, self.bitmask.to(row.device, non_blocking=True))
        t = sample(row[0])
        self.matcher.accept_token(t)
        if self.matcher.is_terminated():  # the grammar took its stop token, <|end_message|>
            self.state = "done"
        return t
