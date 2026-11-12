import json

import pytest

from wsp.action_transport import (
    PROTOCOL_TOKENS,
    THINK_END,
    decode_framed_response,
    framed_response_format,
    parse_json_action,
)


class Tokenizer:
    all_special_tokens = PROTOCOL_TOKENS

    def convert_tokens_to_ids(self, token):
        return 1000 + PROTOCOL_TOKENS.index(token)

    def encode(self, text, **kwargs):
        return (
            [self.convert_tokens_to_ids(text)] if text in PROTOCOL_TOKENS else list(map(ord, text))
        )

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)

    def get_added_vocab(self):
        return {}

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]


def test_framing_keeps_unbounded_reasoning_before_the_action():
    reasoning, boundary, payload = framed_response_format()["format"]["elements"]
    assert reasoning == {"type": "any_tokens", "exclude_tokens": PROTOCOL_TOKENS}
    assert boundary == {"type": "token", "token": THINK_END}
    assert payload["excludes"] == PROTOCOL_TOKENS
    numeric = framed_response_format(Tokenizer())["format"]["elements"]
    assert numeric[1]["token"] == 1001


def test_token_stream_decoding_preserves_escaped_protocol_strings():
    tokenizer = Tokenizer()
    body = "print('</think>')"
    content = json.dumps({"action": "python", "body": body}).replace("<", "\\u003c")
    tokens = (
        list(map(ord, "Reasoning"))
        + [tokenizer.convert_tokens_to_ids(THINK_END)]
        + list(map(ord, content))
        + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    )
    reasoning, decoded = decode_framed_response(tokens, tokenizer)
    assert reasoning == "Reasoning"
    assert parse_json_action(decoded) == ("python", body)


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        {"action": "forward", "body": "{"},
        {"action": "forward", "body": {"type": ["generate"]}},
        {"action": "python", "body": ""},
        {"action": "unknown", "body": "pass"},
        {"action": "python", "body": "pass", "extra": True},
    ],
)
def test_invalid_action_objects_fail_loudly(value):
    with pytest.raises(ValueError):
        parse_json_action(json.dumps(value))


@pytest.mark.parametrize("tokens", [None, [], [True], [42], [1005, 1001, 123]])
def test_incomplete_or_prematurely_ended_frames_are_rejected(tokens):
    with pytest.raises(ValueError):
        decode_framed_response(tokens, Tokenizer())
