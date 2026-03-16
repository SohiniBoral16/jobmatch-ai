"""
LLM Factory — Switch between Claude and OpenAI via .env config.

Usage:
    from utils.llm import get_llm
    llm = get_llm()

Models configured for cost-efficiency (demo-friendly):
    OpenAI  gpt-4o-mini       $0.15 / $0.60  per 1M tokens  (default)
    Claude  claude-haiku-4-5  $1.00 / $5.00  per 1M tokens

Change OPENAI_MODEL or ANTHROPIC_MODEL in .env to upgrade.
"""

import os


def get_llm(temperature: float = 0.0):
    """
    Return a LangChain ChatModel based on the LLM_PROVIDER env var.

    Set in .env:
        LLM_PROVIDER=openai    (recommended — cheaper, default)
        LLM_PROVIDER=claude

    Args:
        temperature: Sampling temperature. 0.0 = deterministic JSON output.

    Returns:
        ChatOpenAI or ChatAnthropic LangChain chat model instance.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    # Remove SOCKS proxy that conflicts with httpx (used by both clients)
    os.environ.pop("ALL_PROXY", None)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in your .env file.\n"
                "Get one at: https://platform.openai.com/api-keys"
            )

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set in your .env file.\n"
                "Get one at: https://console.anthropic.com/settings/keys"
            )

        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            default_headers={"anthropic-version": "2023-06-01"},
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Must be 'openai' or 'claude'."
        )
