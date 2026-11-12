import copy


def transport_messages(messages: list[dict], control_tokens: list[str]) -> list[dict]:
    """Keep literal token spellings as data, not chat-template delimiters."""
    prepared = copy.deepcopy(messages)
    tokens = sorted(set(control_tokens), key=len, reverse=True)
    for message in prepared:
        for field in ["content", "reasoning_content"]:
            value = message.get(field)
            if not isinstance(value, str):
                continue
            for token in tokens:
                if token:
                    replacement = f"\\u{ord(token[0]):04x}" + token[1:]
                    value = value.replace(token, replacement)
            message[field] = value
    return prepared


def build_request(messages: list[dict], editor: dict, prompt_tokens: int) -> dict:
    settings = copy.deepcopy(editor["generation"])
    if settings["reasoning_effort"] != "xhigh":
        raise ValueError("The pinned Qwen editor requires its highest supported effort: xhigh.")
    template = settings["chat_template_kwargs"]
    if (
        template["enable_thinking"] is not True
        or template["preserve_thinking"] is not True
        or template["reasoning_effort"] != "xhigh"
    ):
        raise ValueError("Thinking, preserved thinking, and xhigh must reach the chat template.")
    available = editor["max_model_len"] - prompt_tokens
    if available <= 0:
        raise ValueError("Conversation fills the context window; no output space remains.")
    configured = settings["max_tokens"]
    settings["max_tokens"] = min(configured, available)
    if settings["max_tokens"] != configured:
        print(
            f"Context capacity leaves {available} output tokens, below the configured "
            f"{configured}; no history was silently removed.",
            flush=True,
        )
    return {"model": "wsp-editor", "messages": messages, **settings}
