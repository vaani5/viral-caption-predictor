"""
LLM Rewrite module — uses Groq API to improve posts
based on detected weaknesses from ML model.
"""

from groq import Groq
import os


def rewrite_post(
    original_text: str,
    platform: str,
    score: int,
    weaknesses: list[str],
    api_key: str,
) -> tuple[str, str]:
    """
    Rewrite a post using Groq (Llama 3), guided by specific ML-detected weaknesses.
    Returns (rewritten_text, explanation).
    """

    platform_context = {
        "twitter":   "Twitter/X (280 char limit, casual, punchy, thread-friendly)",
        "reddit":    "Reddit (title-only post, must hook immediately, no fluff)",
        "instagram": "Instagram caption (conversational, emoji-friendly, storytelling tone)",
    }

    weakness_block = "\n".join(f"  • {w}" for w in weaknesses) if weaknesses else "  • General engagement could be stronger"

    prompt = f"""You are an expert social media copywriter who specialises in making posts go viral.

A machine learning model scored this {platform_context.get(platform, platform)} post at {score}/100 for virality potential.

Original post:
"{original_text}"

The ML model detected these specific weaknesses:
{weakness_block}

Your task:
1. Rewrite the post to fix EXACTLY these weaknesses — nothing more.
2. Keep the CORE MESSAGE and meaning completely intact. Do not change what it's about.
3. The rewrite must feel authentic, not clickbait-y or fake.
4. Match the platform's tone: {platform_context.get(platform, platform)}.
5. Do NOT add fictional facts or claims that weren't in the original.

Respond in this exact format:
REWRITE: [your rewritten post here]
EXPLANATION: [one sentence explaining the key change you made]"""

    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # best free model on Groq
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )

    response = chat_completion.choices[0].message.content.strip()

    # Parse response
    rewrite = original_text
    explanation = "Rewrite applied."

    for line in response.split("\n"):
        if line.startswith("REWRITE:"):
            rewrite = line.replace("REWRITE:", "").strip().strip('"')
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()

    return rewrite, explanation


def chat_with_rewriter(
    messages: list[dict],
    context: dict | None,
    api_key: str,
) -> str:
    """
    Conversational rewrite assistant for the floating chatbot widget.

    `messages` — full chat history so far: [{"role": "user"|"assistant", "content": str}, ...]
    `context`  — the currently-scored post (st.session_state.result), or None if
                 nothing has been scored yet. Expected keys: text, platform, score, weaknesses.

    Returns the assistant's plain-text reply.
    """
    if context:
        weakness_block = (
            "\n".join(f"  • {w}" for w in context.get("weaknesses", []))
            or "  • None flagged"
        )
        system_prompt = f"""You are a friendly, sharp social-media copywriting assistant living inside a virality-scoring tool. You chat casually — short, punchy replies, not essays.

The user is currently working on this {context.get('platform', 'post')} post, which an ML model scored {context.get('score', '?')}/100 for virality:
"{context.get('text', '')}"

ML-detected weaknesses on the current version:
{weakness_block}

Help the user rewrite, tighten, or brainstorm the post through conversation — they might ask for a different tone, a shorter version, a platform switch, or just general advice. Whenever you hand back a rewritten version of the post, put it on its own line prefixed with "Here's a rewrite:" followed by the text in quotes, so it's easy to spot and copy. Keep the core meaning intact unless the user explicitly asks you to change it. Never invent facts that weren't in the original."""
    else:
        system_prompt = """You are a friendly, sharp social-media copywriting assistant living inside a virality-scoring tool. You chat casually — short, punchy replies, not essays.

No post has been scored yet, so you don't have ML weaknesses to work from. Help the user brainstorm or draft post ideas conversationally, and nudge them to paste a post and hit "Score this post" first so you can give sharper, data-backed rewrite help."""

    client = Groq(api_key=api_key)

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[-12:]:  # keep recent context only
        api_messages.append({"role": m["role"], "content": m["content"]})

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
        max_tokens=500,
        temperature=0.8,
    )

    return chat_completion.choices[0].message.content.strip()