"""Where the drafter loses acceptance, teacher-forced on fixed text (greedy, EOS respected, prompts outside every
benchmark). Text is generated once by the bf16 target (plain); each profile replays it through verify(1, 8) and the
drafter proposes 7 tokens from every position, so profiles differ only in numerics.
Modal: parity --args "draft_gate0.py [answer tokens] [shortlists]"; with "shortlists" it compares draft shortlists
(/cache/draft_shortlist_mix.bin, from build_shortlist.py mix) on the fp8 + fp4 prefill profile instead, with 16 more non-English prompts.
"""

import gc
import glob
import random
import shutil
import sys

import torch

sys.path.insert(0, "/root")
import inkling_engine.dspark as ds  # noqa: E402
from inkling_engine.engine import Engine  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 32 * 2**30 / total))
tok = AutoTokenizer.from_pretrained(path)
PROMPTS = {
    "chat": ["Explain why the sky looks red at sunset to a ten-year-old.",
             "Write a short, friendly email declining a meeting invitation because of a clash.",
             "What are the trade-offs between renting and buying a home?",
             "Give me a packing list for a four-day hiking trip in cool weather.",
             "Summarise the plot of a made-up detective story set in a lighthouse.",
             "How does a bicycle stay upright while moving?",
             "Compare green tea and black tea: how they are made and how they taste.",
             "Suggest five ways to get better at giving presentations."],
    "code": ["Write a Python function that merges overlapping intervals, with a docstring and three asserts.",
             "Write a C function that reverses a singly linked list in place.",
             "Write a SQL query for the three best-selling products per region from orders(product, region, qty).",
             "Implement an LRU cache class in JavaScript using a Map.",
             "Write a Rust function that counts word frequencies in a string and returns them sorted.",
             "Write a bash script that renames every .jpeg file in a folder to .jpg and prints what it did.",
             "Implement Dijkstra's algorithm in Go with a priority queue.",
             "Write a Python class for a fixed-size ring buffer with push, pop and __len__."],
    "maths": ["Prove that the sum of the first n odd numbers is n squared.",
              "Solve x^2 - 7x + 10 = 0 and check both roots.",
              "A bag has 5 red and 7 blue marbles. Two are drawn without replacement. Find the chance both are red.",
              "Differentiate f(x) = x^3 * ln(x) and simplify.",
              "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the steps.",
              "Find the determinant of the matrix [[2, 1, 3], [0, 4, 1], [5, 2, 0]].",
              "Show that the square root of 2 is irrational.",
              "Compute the sum of the geometric series 3 + 6 + 12 + ... up to 10 terms."],
    "other": ["Explique en français comment fonctionne un moteur à quatre temps.",
              "Erkläre auf Deutsch, warum es Ebbe und Flut gibt.",
              "Escribe en español una receta sencilla de sopa de verduras.",
              "用中文简要介绍长城的历史。",
              "日本語で、東京から京都への行き方を説明してください。",
              "Explique em português o que é inflação.",
              "Объясни по-русски, как работает холодильник.",
              "اشرح بالعربية كيف تتكون الغيوم."],
}
src = [open(f, encoding="utf-8").read() for f in sorted(glob.glob("/root/inkling_engine/*.py")
                                                       + glob.glob("/root/bench/*.py"))]
SHORTLISTS = len(sys.argv) > 3 and sys.argv[3] == "shortlists"
if SHORTLISTS:
    PROMPTS["other"] += [
        "Racconta in italiano la leggenda di Romolo e Remo in breve.", "Leg in het Nederlands uit hoe een vulkaan werkt.",
        "Opowiedz po polsku, jak powstaje miód.", "Beskriv på svenska hur man bakar kanelbullar.",
        "한국어로 세종대왕에 대해 짧게 설명해 주세요.", "हिन्दी में बताइए कि बारिश कैसे होती है।",
        "Türkçe olarak çayın nasıl demlendiğini anlat.", "Tiếng Việt: giải thích quang hợp là gì.",
        "ไทย: อธิบายวิธีทำข้าวผัดอย่างง่าย", "Explique em francês como funciona uma bateria.",
        "Schreibe auf Deutsch einen kurzen Brief an die Nachbarn wegen einer Feier.",
        "Escribe en español un resumen de cómo se forman los huracanes.", "Напиши по-русски короткую сказку о лисе.",
        "用中文写一个关于勤奋的小故事。", "日本語で、健康的な朝食の例を五つ挙げてください。",
        "اكتب بالعربية فقرة عن أهمية النوم."]
rng = random.Random(0)
PROMPTS["long"] = []
for q in ["What does this code do overall?", "List the classes defined here and their purpose.",
          "Point out two places that could be simplified.", "Which functions allocate GPU buffers?",
          "Describe the control flow of the main entry point.", "What configuration does it read?",
          "Write a short README section for it.", "What tests would you add, and why?"]:
    PROMPTS["long"].append("\n\n".join(x[:9000] for x in rng.sample(src, 2)) + "\n\n" + q)


def encode(text):
    r = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    return list(r["input_ids"] if hasattr(r, "keys") else r)


items = [(t, encode(p)) for t, ps in PROMPTS.items() for p in ps]


@torch.inference_mode()
def replay(eng, ids, text):
    """Per position of `text` with 7 labels left: (accepted, [label in the candidates] * 7); and target agreement."""
    d = eng.drafter
    lg, _ = eng.prefill(ids)
    P, rows = len(ids), []
    n_agree, n_tgt = int(int(lg.argmax()) == text[0]), 1
    for b in range(0, len(text) - 8, 8):
        part = text[b:b + 8]
        logits, taps = eng.verify[(1, 8)].run(torch.tensor(part, device="cuda"), [0], [P + b])
        arg = logits.argmax(-1).tolist()
        n_agree += sum(int(arg[j] == text[b + j + 1]) for j in range(8))
        n_tgt += 8
        taps = taps[:8].clone()
        eng.cache.commit_spec(0, 8, 0)
        for j in range(8):
            toks, _ = d.draft(text[b + j], P + b + j, 0.0, slot=0)
            lab = text[b + j + 1:b + j + 8]
            if len(lab) == 7:
                t = torch.tensor(lab, device="cuda")
                inc = (d.cand_buf == t[:, None]).any(1).tolist()
                dr = toks.tolist()
                rows.append((next((i for i in range(7) if dr[i] != lab[i]), 7), inc))
            d.commit(taps[j:j + 1], P + b + j, slot=0)
    return rows, n_agree, n_tgt


def stats(rows):
    n, acc = len(rows), [a for a, _ in rows]
    miss = [(a, inc) for a, inc in rows if a < 7]
    return dict(n=n, p=sum(acc) / max(sum(min(a + 1, 7) for a in acc), 1),  # accepted / attempted drafts
                a4=sum(min(a, 3) for a in acc) / n + 1, a8=sum(acc) / n + 1,
                hidden=sum(1 - int(inc[a]) for a, inc in miss) / max(len(miss), 1))


texts, res = {}, {}


def make(name, shortlist=32768, **kw):
    ds.SHORTLIST = shortlist
    eng = Engine(path, max_ctx=16384, spec=True, slots=1, **kw)
    print(f"== {name}: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return eng


def free(eng):
    eng.__dict__.clear()
    gc.collect()
    torch.cuda.empty_cache()


def gen(eng, key):
    stop = {eng.cfg.eos_token_id}
    texts[key] = []
    for _, ids in items:
        out, _ = eng.generate(ids, N, stop, spec=False)
        texts[key].append(out[:-1] if out[-1] in stop else out)


def run(name, eng, key):
    per, ag, tg, by = [], 0, 0, {}
    for (t, ids), text in zip(items, texts[key]):
        if len(text) < 24:
            per.append([])
            continue
        rows, a, n = replay(eng, ids, text)
        per.append(rows)
        ag, tg = ag + a, tg + n
        by.setdefault(t, []).extend(rows)
    res[(name, key)] = per
    s = stats([r for rows in per for r in rows])
    print(f"{name:18s} text {key}: n {s['n']}, p {s['p']:.4f}, accept len width 4 {s['a4']:.3f} / 8 {s['a8']:.3f}, misses "
          f"with the token outside the 1024 candidates {s['hidden']:.3f}, target == text {ag / tg:.3f}")
    print("   " + " | ".join(f"{t} p {stats(r)['p']:.3f} a4 {stats(r)['a4']:.2f} n {len(r)}" for t, r in by.items()),
          flush=True)


def paired(x, y):
    """p(x) - p(y) over the same prompts, with a bootstrap 95% interval over prompts."""
    ax, ay = res[x], res[y]
    idx = [i for i in range(len(ax)) if ax[i] and ay[i]]

    def diff(ii):
        return stats([r for i in ii for r in ax[i]])["p"] - stats([r for i in ii for r in ay[i]])["p"]

    g = random.Random(1)
    bs = sorted(diff([g.choice(idx) for _ in idx]) for _ in range(1000))
    print(f"paired p{x} - p{y} = {diff(idx):+.4f}, 95% [{bs[25]:+.4f}, {bs[974]:+.4f}]", flush=True)


if SHORTLISTS:
    # the drafter reads its packaged list: swap the file in this container (the engine is unchanged)
    data = ds.Path(ds.__file__).parent / "data" / "draft_shortlist.bin"
    shutil.copy(data, "/tmp/en.bin")
    files = {"en": "/tmp/en.bin", "mix": "/cache/draft_shortlist_mix.bin"}
    variants = [("en 32k", "en", 32768), ("mix 32k", "mix", 32768), ("en 48k", "en", 49152), ("mix 48k", "mix", 49152),
                ("mix 64k", "mix", 65536)]
    for name, file, rows in variants:
        shutil.copy(files[file], data)
        eng = make(name, shortlist=rows, dense_fp8=True, fp4_prefill=True)
        if "own" not in texts:
            gen(eng, "own")
        sl = set(eng.drafter.short_ids.tolist())
        n_out = sum(x not in sl for (c, _), ts in zip(items, texts["own"]) for x in ts if c == "other")
        n_oth = sum(len(ts) for (c, _), ts in zip(items, texts["own"]) if c == "other")
        print(f"{name}: other-language text tokens outside the shortlist {n_out / n_oth:.4f}")
        t0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g = eng.drafter.graphs[(eng.drafter.buckets[0], 0)]
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        t0[0].record()
        for _ in range(200):
            g.replay()
        t0[1].record()
        torch.cuda.synchronize()
        print(f"{name}: draft graph {t0[0].elapsed_time(t0[1]) / 200:.3f} ms")
        run(name, eng, "own")
        free(eng)
        del eng
    for name, _, _ in variants[1:]:
        paired((name, "own"), ("en 32k", "own"))
        for t in PROMPTS:
            x, y = res[(name, "own")], res[("en 32k", "own")]
            ii = [i for i, (c, _) in enumerate(items) if c == t and x[i] and y[i]]
            g = random.Random(2)

            def diff(js):
                return stats([r for j in js for r in x[j]])["p"] - stats([r for j in js for r in y[j]])["p"]

            bs = sorted(diff([g.choice(ii) for _ in ii]) for _ in range(1000))
            print(f"   {name} - en 32k, {t}: {diff(ii):+.4f} [{bs[25]:+.4f}, {bs[974]:+.4f}]", flush=True)
    sys.exit(0)
eng = make("bf16")
gen(eng, "bf16")
flat = [t for ts in texts["bf16"] for t in ts]
sl = set(eng.drafter.short_ids.tolist())
print(f"answers {len(texts['bf16'])}, tokens {len(flat)}, shorter than 24: {sum(len(t) < 24 for t in texts['bf16'])}, "
      f"prompt tokens max {max(len(i) for _, i in items)}")
print(f"text tokens outside the 32k shortlist: {sum(t not in sl for t in flat) / len(flat):.4f}")
for t in PROMPTS:
    tt = [x for (c, _), ts in zip(items, texts["bf16"]) for x in ts if c == t]
    print(f"   {t}: {len(tt)} tokens, outside the shortlist {sum(x not in sl for x in tt) / max(len(tt), 1):.4f}")
run("bf16", eng, "bf16")
free(eng)
del eng
eng = make("bf16 full vocab", shortlist=0)
run("bf16 full vocab", eng, "bf16")
free(eng)
del eng
eng = make("fp8 dense", dense_fp8=True)
run("fp8 dense", eng, "bf16")
free(eng)
del eng
eng = make("fp8 + fp4 prefill", dense_fp8=True, fp4_prefill=True)
run("fp8 + fp4 prefill", eng, "bf16")
gen(eng, "own")
run("fp8 + fp4 prefill", eng, "own")
base = ("bf16", "bf16")
paired(base, ("bf16 full vocab", "bf16"))
paired(base, ("fp8 dense", "bf16"))
paired(base, ("fp8 + fp4 prefill", "bf16"))
