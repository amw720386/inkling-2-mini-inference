from inkling_engine.server import Splitter


def run(pieces):
    s = Splitter()
    r, c = [], []
    for i, p in enumerate(pieces):
        a, b = s.feed(p, final=i == len(pieces) - 1)
        r.append(a)
        c.append(b)
    return "".join(r), "".join(c)


def test_reasoning_then_content():
    txt = "<|message_model|><|content_thinking|>think hard<|end_message|><|message_model|><|content_text|>Paris.<|content_model_end_sampling|>"
    assert run([txt]) == ("think hard", "Paris.")


def test_split_inside_special_token():
    pieces = ["<|message_model|><|con", "tent_thinking|>ab", "c<|end_mes", "sage|><|content_text|>x", "y"]
    assert run(pieces) == ("abc", "xy")


def test_plain_text_and_unknown_tokens():
    assert run(["hello <|unused_1|>world"]) == ("", "hello world")
