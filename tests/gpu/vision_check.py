"""Image input: engine vs transformers Inkling2LoopForConditionalGeneration on a synthetic image.

Modal: modal run modal/engine_app.py::parity --args "vision_check.py"   (no memory cap: both models load)
"""

import gc
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root")
from inkling_engine.engine import Engine  # noqa: E402

path = sys.argv[1]
img = Image.new("RGB", (480, 320), (220, 30, 30))
d = ImageDraw.Draw(img)
try:
    font = ImageFont.load_default(size=160)
except TypeError:
    font = ImageFont.load_default()
d.text((140, 60), "42", fill=(20, 40, 230), font=font)
messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "What color is the background, and what number is written? Answer briefly."}]}]

from transformers import AutoProcessor, Inkling2LoopForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(path)
enc = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                               return_tensors="pt", reasoning_effort="none")
print("prompt tokens", enc["input_ids"].shape[1], "image tokens", int((enc["input_ids"] == 200054).sum()))
hf = Inkling2LoopForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
with torch.no_grad():
    inp = {k: v.to("cuda") for k, v in enc.items()}
    hf_logits = hf(**inp).logits[0, -1].float().cpu()
    hf_out = hf.generate(**inp, max_new_tokens=40, do_sample=False)[0, enc["input_ids"].shape[1]:].tolist()
del hf
gc.collect()
torch.cuda.empty_cache()
print("hf:", repr(proc.tokenizer.decode(hf_out)))

eng = Engine(path, max_ctx=4096, spec=True, vision=True)
ids, feats = eng.vision.chat_inputs(messages, reasoning_effort="none")
assert ids == enc["input_ids"][0].tolist()
logits, _ = eng.prefill(ids, feats)
a, b = torch.log_softmax(logits.float().cpu(), -1), torch.log_softmax(hf_logits, -1)
print(f"first-token KL(hf || engine) = {(b.exp() * (b - a)).sum().item():.2e}, argmax "
      f"{'same' if logits.argmax().item() == hf_logits.argmax().item() else 'DIFF'}")
out, st = eng.generate(ids, 40, {eng.cfg.eos_token_id}) if False else (None, None)
toks = list(eng.stream(ids, 40, {eng.cfg.eos_token_id}, image_feats=feats))
text = proc.tokenizer.decode(toks)
print("engine:", repr(text))
same = next((i for i, (x, y) in enumerate(zip(toks, hf_out)) if x != y), min(len(toks), len(hf_out)))
print(f"greedy match {same}/{len(hf_out)}")
ok = "red" in text.lower() and "42" in text
print("VISION", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
