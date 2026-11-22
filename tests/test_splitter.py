from inkling_engine.server import Splitter, normalize_messages


def run(pieces):
    s = Splitter()
    r, c, calls = [], [], []
    for i, p in enumerate(pieces):
        a, b, k = s.feed(p, final=i == len(pieces) - 1)
        r.append(a)
        c.append(b)
        calls += k
    return "".join(r), "".join(c), calls


def test_reasoning_then_content():
    txt = "<|message_model|><|content_thinking|>think hard<|end_message|><|message_model|><|content_text|>Paris.<|content_model_end_sampling|>"
    assert run([txt]) == ("think hard", "Paris.", [])


def test_split_inside_special_token():
    pieces = ["<|message_model|><|con", "tent_thinking|>ab", "c<|end_mes", "sage|><|content_text|>x", "y"]
    assert run(pieces) == ("abc", "xy", [])


def test_plain_text_and_unknown_tokens():
    assert run(["hello <|unused_1|>world"]) == ("", "hello world", [])


def test_tool_call():
    txt = ('<|message_model|><|content_thinking|>need weather<|end_message|>'
           '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}'
           '<|end_message|><|content_model_end_sampling|>')
    r, c, calls = run([txt[:70], txt[70:120], txt[120:]])
    assert r == "need weather" and c == ""
    assert calls == [("get_weather", {"city": "Paris"})]


def test_two_tool_calls_streamed_char_by_char():
    txt = ('<|message_model|>a<|content_invoke_tool_json|>{"name":"a","args":{}}<|end_message|>'
           '<|message_model|>b<|content_invoke_tool_json|>{"name":"b","args":{"x":[1,2]}}<|end_message|>')
    _, c, calls = run(list(txt))
    assert c == "" and calls == [("a", {}), ("b", {"x": [1, 2]})]


def test_normalize_string_arguments():
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]}]
    out = normalize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert out[0]["content"] == ""
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # input untouched
