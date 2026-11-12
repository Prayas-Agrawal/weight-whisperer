"""Action framing that leaves reasoning free while requiring a complete final payload."""

import copy
import json

THINK_END = "</think>"
PROTOCOL_TOKENS = [
    "<think>",
    THINK_END,
    "<tool_call>",
    "</tool_call>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]

ACTION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["python", "final"]},
                "body": {"type": "string"},
            },
            "required": ["action", "body"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "forward"},
                "body": {
                    "type": "object",
                    "properties": {"type": {"enum": ["generate", "teacher_forced"]}},
                    "required": ["type"],
                    "additionalProperties": True,
                },
            },
            "required": ["action", "body"],
            "additionalProperties": False,
        },
    ],
}


def protocol_token_ids(tokenizer) -> dict[str, int]:
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in PROTOCOL_TOKENS}
    if any(type(value) is not int for value in ids.values()):
        raise ValueError("The configured tokenizer lacks required framing tokens.")
    for token, token_id in ids.items():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError("A protocol boundary is not one dedicated tokenizer token.")
    return ids


def framed_response_format(tokenizer=None) -> dict:
    tokens = (
        protocol_token_ids(tokenizer)
        if tokenizer is not None
        else {token: token for token in PROTOCOL_TOKENS}
    )
    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "any_tokens", "exclude_tokens": list(tokens.values())},
                {"type": "token", "token": tokens[THINK_END]},
                {
                    "type": "json_schema",
                    "json_schema": copy.deepcopy(ACTION_SCHEMA),
                    "excludes": list(PROTOCOL_TOKENS),
                },
            ],
        },
    }


def parse_json_action(text: str) -> tuple[str, str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Return one complete JSON object with action and body fields.") from error
    if not isinstance(value, dict) or set(value) != {"action", "body"}:
        raise ValueError("Action JSON must contain exactly action and body.")
    if not isinstance(value["action"], str) or value["action"] not in {
        "python",
        "forward",
        "final",
    }:
        raise ValueError("Action must be python, forward or final.")
    if value["action"] == "forward":
        if not isinstance(value["body"], dict):
            raise ValueError("Forward body must be a request object, not serialized JSON text.")
        kind = value["body"].get("type")
        if not isinstance(kind, str) or kind not in {"generate", "teacher_forced"}:
            raise ValueError("Forward body must specify generate or teacher_forced.")
        return "forward", json.dumps(value["body"], ensure_ascii=False)
    if not isinstance(value["body"], str) or not value["body"].strip():
        raise ValueError("Action body must be a nonempty string.")
    return value["action"], value["body"]


def decode_framed_response(token_ids: object, tokenizer) -> tuple[str, str]:
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(i) is not int for i in token_ids)
    ):
        raise ValueError("Framed action responses require the actual generated token IDs.")
    ids = protocol_token_ids(tokenizer)
    try:
        boundary = token_ids.index(ids[THINK_END])
    except ValueError as error:
        raise ValueError(
            "The response did not contain the required reasoning/action boundary."
        ) from error
    protected = set(ids.values())
    if any(token in protected for token in token_ids[:boundary]):
        raise ValueError("A protocol terminator appeared before the action boundary.")
    content_ids = token_ids[boundary + 1 :]
    stops = {ids["<|im_end|>"], ids["<|endoftext|>"]}
    if content_ids and content_ids[-1] in stops:
        content_ids = content_ids[:-1]
    if any(token in protected for token in content_ids):
        raise ValueError("An unescaped protocol token appeared inside the action payload.")
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    reasoning = tokenizer.decode(token_ids[:boundary], **kwargs)
    content = tokenizer.decode(content_ids, **kwargs)
    parse_json_action(content)
    return reasoning, content
