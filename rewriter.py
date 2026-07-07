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