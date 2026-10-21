"""Shared custom-mode UI and token budget for vision-language nodes."""

CUSTOM_TOKEN_LIMIT = 32_000
CUSTOM_SYSTEM_PROMPT_INPUT = ("STRING", {
    "default": "",
    "multiline": True,
    "tooltip": "Used only with the custom preset. Sent directly as the system message; prompt is the user message. No preset or camera instructions are added. Input (including media) and output share a 32,000-token budget.",
})


def vl_preset_options(presets):
    """Keep custom available even when local preset overrides replace the list."""
    return list(dict.fromkeys([*(presets or ["IMG › Detailed"]), "custom"]))


def remaining_custom_tokens(input_tokens, requested_tokens, limit=CUSTOM_TOKEN_LIMIT):
    remaining = int(limit) - int(input_tokens)
    if remaining <= 0:
        raise ValueError(f"Custom input uses {input_tokens} tokens; it must leave room for output within the {limit:,}-token total limit.")
    return min(int(requested_tokens), remaining)
