"""
Viral Post Predictor — Streamlit App
Light/dark, gradient-accented, GenZ-leaning UI + floating rewrite chatbot.
"""

import streamlit as st
import plotly.graph_objects as go
import os, sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import load_model, predict, get_weaknesses, extract_features
from rewriter import rewrite_post, chat_with_rewriter

st.set_page_config(
    page_title="Viral or Spiral?",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme setup ───────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # GenZ default: dark + neon gradient

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

THEMES = {
    "light": {
        "bg": "#F5F2EE", "panel": "#FFFFFF", "border": "#E0DAD2", "border_soft": "#F0EBE4",
        "text": "#1A1815", "text_soft": "#4A4540", "text_mute": "#7A7268",
        "text_dim": "#B0A898", "text_faint": "#C0B8B0",
        "input_bg": "#FAFAF8",
        "accent": "#C8873A", "accent2": "#E85D8A", "accent3": "#7C5CFA",
        "accent_hover": "#B07530",
        "success": "#1A7A46", "danger": "#C0441C", "warn": "#9A6B10",
        "shadow": "0 2px 12px rgba(0,0,0,0.05)",
        "glow": "0 4px 18px rgba(200,135,58,0.22)",
    },
    "dark": {
        "bg": "#0A0A10", "panel": "#15151F", "border": "#2A2A3B", "border_soft": "#20202C",
        "text": "#F2F0F8", "text_soft": "#CBC7DC", "text_mute": "#9490AC",
        "text_dim": "#6C6884", "text_faint": "#524E6C",
        "input_bg": "#1B1B28",
        "accent": "#B76BFF", "accent2": "#FF5FA2", "accent3": "#5CE1FF",
        "accent_hover": "#C784FF",
        "success": "#4ADE80", "danger": "#FF6B81", "warn": "#FBBF24",
        "shadow": "0 4px 20px rgba(0,0,0,0.35)",
        "glow": "0 0 24px rgba(183,107,255,0.35)",
    },
}
T = THEMES[st.session_state.theme]
GRADIENT = f"linear-gradient(120deg, {T['accent']} 0%, {T['accent2']} 55%, {T['accent3']} 100%)"


def build_css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body {{ background: {t['bg']} !important; }}
.stApp, .stApp > div, [class*="css"],
section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"],
div[data-testid="stHeader"],
div[data-testid="block-container"],
div.main, div.main > div {{
  background: {t['bg']} !important;
  color: {t['text']} !important;
  font-family: 'Inter', sans-serif !important;
}}
div[data-testid="stAppViewContainer"], div[data-testid="stMain"] {{
  overflow: visible !important;
}}

div[data-testid="column"] {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
div[data-testid="column"] > div {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
div[data-testid="stVerticalBlock"] > div[style*="background"] {{ background: transparent !important; }}
div.element-container {{ background: transparent !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 2rem 4rem !important; max-width: 1280px !important; }}

/* HERO */
.hero {{
  padding: 3rem 0 2rem;
  border-bottom: 1.5px solid {t['border']};
  margin-bottom: 2rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
}}
.hero-eyebrow {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: {t['accent']};
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}}
.hero-title {{
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 4.5vw, 3.8rem);
  font-weight: 800;
  line-height: 1.0;
  color: {t['text']};
  margin-bottom: 0.55rem;
}}
.hero-title .accent {{
  background: {GRADIENT};
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.hero-sub {{ font-size: 14px; color: {t['text_mute']}; line-height: 1.65; max-width: 420px; }}
.hero-meta {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: {t['text_dim']};
  text-align: right;
  line-height: 2;
}}

/* PANEL */
.panel {{
  background: {t['panel']};
  border: 1.5px solid {t['border']};
  border-radius: 20px;
  padding: 1.5rem;
  box-shadow: {t['shadow']};
}}
.panel-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: {t['text_dim']};
  margin-bottom: 1.1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid {t['border_soft']};
}}

/* TEXTAREA */
.stTextArea textarea {{
  background: {t['input_bg']} !important;
  border: 1.5px solid {t['border']} !important;
  border-radius: 12px !important;
  color: {t['text']} !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
}}
.stTextArea textarea:focus {{
  border-color: {t['accent']} !important;
  box-shadow: 0 0 0 3px {_rgba(t['accent'], 0.16)} !important;
}}
.stTextArea label {{ display: none !important; }}

/* SELECTBOX */
.stSelectbox > div > div {{
  background: {t['input_bg']} !important;
  border: 1.5px solid {t['border']} !important;
  border-radius: 12px !important;
  color: {t['text']} !important;
  font-size: 13px !important;
}}
.stSelectbox label {{ display: none !important; }}

/* BUTTONS */
.stButton > button {{
  width: 100% !important;
  border-radius: 99px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 0.65rem 1.5rem !important;
  border: none !important;
  transition: all 0.18s ease !important;
}}
.stButton > button:not(:disabled) {{
  background: {GRADIENT} !important;
  color: #FFFFFF !important;
}}
.stButton > button:not(:disabled):hover {{
  filter: brightness(1.08) !important;
  transform: translateY(-1px) !important;
  box-shadow: {t['glow']} !important;
}}
.stButton > button:disabled {{
  background: {t['border_soft']} !important;
  color: {t['text_faint']} !important;
  cursor: not-allowed !important;
}}

/* TEXT INPUT */
.stTextInput input {{
  background: {t['input_bg']} !important;
  border: 1.5px solid {t['border']} !important;
  border-radius: 10px !important;
  color: {t['text']} !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
}}
.stTextInput input:focus {{ border-color: {t['accent']} !important; }}
.stTextInput label {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: {t['text_dim']} !important;
}}

/* CHAT INPUT (native st.chat_input, reused in floating panel) */
.stChatInput textarea, div[data-testid="stChatInput"] textarea {{
  background: {t['input_bg']} !important;
  color: {t['text']} !important;
  border: 1.5px solid {t['border']} !important;
  border-radius: 14px !important;
}}
div[data-testid="stChatInput"] {{ background: transparent !important; }}
div[data-testid="stChatInput"] button {{
  background: {GRADIENT} !important;
  border-radius: 99px !important;
}}

/* CHAR COUNT */
.char-count {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  text-align: right;
  margin-top: -8px;
  margin-bottom: 10px;
  color: {t['text_dim']};
}}

/* VERDICT */
.verdict-wrap {{ text-align: center; margin: 0.5rem 0 1rem; }}
.verdict-badge {{
  display: inline-block;
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  padding: 5px 16px;
  border-radius: 99px;
  letter-spacing: 0.04em;
}}
.v-low    {{ background: {_rgba(t['danger'], 0.12)}; color: {t['danger']}; border: 1.5px solid {_rgba(t['danger'], 0.35)}; }}
.v-medium {{ background: {_rgba(t['warn'], 0.12)}; color: {t['warn']}; border: 1.5px solid {_rgba(t['warn'], 0.35)}; }}
.v-high   {{ background: {_rgba(t['success'], 0.12)}; color: {t['success']}; border: 1.5px solid {_rgba(t['success'], 0.35)}; }}

/* FEATURE BARS */
.feat-section-title {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {t['text_dim']};
  margin: 1.25rem 0 0.75rem;
  padding-top: 1rem;
  border-top: 1px solid {t['border_soft']};
}}
.feat-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.feat-name {{
  font-size: 12px; color: {t['text_soft']}; width: 145px; flex-shrink: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.feat-track {{ flex: 1; height: 5px; background: {t['border_soft']}; border-radius: 99px; overflow: hidden; }}
.feat-fill {{ height: 5px; border-radius: 99px; }}
.fill-pos {{ background: linear-gradient(90deg, {t['success']}, {t['accent3']}); }}
.fill-neg {{ background: linear-gradient(90deg, {t['danger']}, {t['warn']}); }}
.feat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; width: 38px; text-align: right; flex-shrink: 0; }}
.val-pos {{ color: {t['success']}; }}
.val-neg {{ color: {t['danger']}; }}

/* WEAKNESS */
.weakness-wrap {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {t['border_soft']}; }}
.weakness-title {{
  font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: {t['text_dim']}; margin-bottom: 0.6rem;
}}
.weakness-item {{
  font-size: 12px; color: {t['text_soft']}; padding: 7px 10px 7px 12px;
  border-left: 2.5px solid {t['accent']}; margin-bottom: 6px; line-height: 1.55;
  background: {_rgba(t['accent'], 0.08)}; border-radius: 0 8px 8px 0;
}}

/* EMPTY STATE */
.empty-state {{ text-align: center; padding: 3.5rem 1rem; }}
.empty-icon {{ font-size: 2.2rem; margin-bottom: 0.75rem; }}
.empty-text {{ font-size: 13px; color: {t['text_faint']}; line-height: 1.7; }}

/* REWRITE */
.rewrite-scores {{ display: flex; gap: 8px; margin-bottom: 1rem; }}
.rscore-box {{ flex: 1; text-align: center; padding: 10px 8px; border-radius: 12px; }}
.rscore-before {{ background: {_rgba(t['danger'], 0.10)}; border: 1.5px solid {_rgba(t['danger'], 0.3)}; }}
.rscore-after  {{ background: {_rgba(t['success'], 0.10)}; border: 1.5px solid {_rgba(t['success'], 0.3)}; }}
.rscore-delta  {{ background: {t['input_bg']}; border: 1.5px solid {t['border']}; }}
.rscore-label {{
  font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 0.12em;
  text-transform: uppercase; color: {t['text_dim']}; margin-bottom: 4px;
}}
.rscore-val {{ font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; }}
.rv-before {{ color: {t['danger']}; }}
.rv-after  {{ color: {t['success']}; }}
.rv-delta-pos {{ color: {t['success']}; }}
.rv-delta-neg {{ color: {t['danger']}; }}
.rv-delta-neu {{ color: {t['text_mute']}; }}

.rewrite-text {{
  background: {_rgba(t['success'], 0.08)};
  border: 1.5px solid {_rgba(t['success'], 0.3)};
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
  font-size: 14px;
  color: {t['text']};
  line-height: 1.75;
  margin-bottom: 0.75rem;
}}
.rewrite-explanation {{ font-size: 11px; color: {t['success']}; font-style: italic; padding-left: 4px; line-height: 1.5; }}

/* PLATFORM HINT */
.platform-hint {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {t['text_dim']}; margin-top: 6px; margin-bottom: 8px; padding-left: 2px; }}

/* TIPS BOX */
.tips-box {{ background: {t['input_bg']}; border: 1.5px solid {t['border']}; border-radius: 14px; padding: 1rem 1.1rem; margin-top: 1rem; }}
.tips-title {{
  font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: {t['text_dim']}; margin-bottom: 0.5rem;
}}
.tips-body {{ font-size: 12px; color: {t['text_mute']}; line-height: 2; }}

/* API KEY LOADED */
.api-loaded {{
  font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {t['success']};
  margin-bottom: 0.75rem; background: {_rgba(t['success'], 0.12)}; padding: 6px 10px;
  border-radius: 6px; border: 1px solid {_rgba(t['success'], 0.3)};
}}

/* EXPANDER */
.streamlit-expanderHeader {{
  background: {t['input_bg']} !important; border-radius: 10px !important; font-size: 12px !important;
  color: {t['text_mute']} !important; border: 1px solid {t['border']} !important;
}}

hr {{ border-color: {t['border']} !important; }}

/* ── THEME TOGGLE (fixed, top-right) ─────────────────────────────────── */
div[class*="st-key-theme_toggle_container"] {{
  position: fixed !important;
  top: 18px; right: 24px;
  z-index: 9998;
  width: 46px; height: 46px;
}}
div[class*="st-key-theme_toggle_container"] .stButton > button {{
  width: 46px !important; height: 46px !important; padding: 0 !important;
  border-radius: 50% !important; font-size: 18px !important;
  background: {t['panel']} !important; border: 1.5px solid {t['border']} !important;
  color: {t['text']} !important; box-shadow: {t['shadow']} !important;
}}
div[class*="st-key-theme_toggle_container"] .stButton > button:hover {{
  transform: translateY(-1px) scale(1.04) !important;
  box-shadow: {t['glow']} !important;
}}

/* ── FLOATING CHAT FAB ────────────────────────────────────────────────── */
div[class*="st-key-fab_container"] {{
  position: fixed !important;
  bottom: 24px; right: 24px;
  z-index: 9999;
  width: 60px; height: 60px;
}}
div[class*="st-key-fab_container"] .stButton > button {{
  width: 60px !important; height: 60px !important; padding: 0 !important;
  border-radius: 50% !important; font-size: 24px !important;
  background: {GRADIENT} !important; border: none !important;
  box-shadow: {t['glow']} !important;
}}
div[class*="st-key-fab_container"] .stButton > button:hover {{
  transform: scale(1.06) !important;
}}

/* ── FLOATING CHAT PANEL ──────────────────────────────────────────────── */
div[class*="st-key-chat_panel"] {{
  position: fixed !important;
  bottom: 96px; right: 24px;
  z-index: 9999;
  width: 360px;
  max-width: 90vw;
  background: {t['panel']};
  border: 1.5px solid {t['border']};
  border-radius: 20px;
  box-shadow: {t['shadow']}, {t['glow']};
  padding: 1rem 1rem 0.6rem;
}}
.chat-header {{
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 15px;
  color: {t['text']}; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 6px;
}}
.chat-sub {{
  font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {t['text_dim']};
  margin-bottom: 0.75rem; line-height: 1.5;
}}
div[class*="st-key-chat_scroll"] {{
  border-radius: 12px;
  background: {t['input_bg']};
  border: 1px solid {t['border_soft']};
  padding: 8px;
}}
.chat-empty {{ font-size: 12px; color: {t['text_faint']}; text-align: center; padding: 1.5rem 0.5rem; line-height: 1.6; }}
.chat-bubble {{
  font-size: 13px; line-height: 1.55; padding: 8px 12px; border-radius: 14px;
  margin-bottom: 8px; max-width: 88%; word-wrap: break-word; white-space: pre-wrap;
}}
.chat-user {{
  background: {GRADIENT}; color: #FFFFFF; margin-left: auto; border-bottom-right-radius: 4px;
}}
.chat-bot {{
  background: {t['panel']}; border: 1px solid {t['border']}; color: {t['text']};
  margin-right: auto; border-bottom-left-radius: 4px;
}}
</style>
"""

st.markdown(build_css(T), unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    with st.spinner("Warming up the algorithm..."):
        st.session_state.model, st.session_state.scaler = load_model()

if "result"         not in st.session_state: st.session_state.result = None
if "rewrite_result" not in st.session_state: st.session_state.rewrite_result = None
if "chat_open"       not in st.session_state: st.session_state.chat_open = False
if "chat_messages"   not in st.session_state: st.session_state.chat_messages = []


# ── Theme toggle button (fixed, top-right) ──────────────────────────────────
with st.container(key="theme_toggle_container"):
    toggle_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    if st.button(toggle_icon, key="theme_toggle_btn"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# ── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-eyebrow">⚡ Virality Intelligence · ML + GenAI</div>
    <div class="hero-title"><span class="accent">Viral or Spiral?</span></div>
    <div class="hero-sub">Paste your post. Get a virality score, a signal-by-signal breakdown, and an AI rewrite that fixes exactly what's weak.</div>
  </div>
  
</div>
""", unsafe_allow_html=True)


# ── Platform config ─────────────────────────────────────────────────────────
PLATFORM_HINTS = {
    "twitter":   "60–220 chars · 1–2 hashtags · hooks + questions win",
    "reddit":    "Punchy title under 120 chars · no hashtags · no emojis",
    "instagram": "80–300 chars · 5–10 hashtags · CTA + emojis rewarded",
}
PLATFORM_ICONS = {"twitter": "✖️", "reddit": "🤖", "instagram": "📸"}
EXAMPLES = {
    "twitter":   "I quit my job 2 years ago to build in public. Here's everything nobody told me 🧵",
    "reddit":    "What's the one thing you wish someone told you before your first dev job?",
    "instagram": "Nobody posts the 2am debugging sessions. I'm posting it. 3 years of failing before my first dev job 💙 Save this for when you feel like quitting #coding #developerlife #100DaysOfCode",
}
LIMITS = {"twitter": 280, "reddit": 300, "instagram": 2200}


# ── THREE COLUMNS ──────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1.05, 1.0, 1.05], gap="medium")


# ╔══════════════════╗
# ║  LEFT — Input    ║
# ╚══════════════════╝
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Your post</div>', unsafe_allow_html=True)

    platform = st.selectbox(
        "platform",
        ["twitter", "reddit", "instagram"],
        format_func=lambda x: f"{PLATFORM_ICONS[x]}  {x.capitalize()}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="platform-hint">{PLATFORM_HINTS[platform]}</div>',
        unsafe_allow_html=True,
    )

    post_text = st.text_area(
        "post",
        value=EXAMPLES[platform],
        height=185,
        label_visibility="collapsed",
        placeholder="What are you about to post?",
    )

    char_len = len(post_text)
    limit    = LIMITS[platform]
    cc_color = T["danger"] if char_len > limit else T["text_dim"]
    st.markdown(
        f'<div class="char-count" style="color:{cc_color};">{char_len} / {limit}</div>',
        unsafe_allow_html=True,
    )

    if st.button("⚡  Score this post", key="score_btn"):
        if post_text.strip():
            with st.spinner("Reading the room..."):
                score, breakdown, feats = predict(
                    post_text, platform,
                    st.session_state.model,
                    st.session_state.scaler,
                )
                weaknesses = get_weaknesses(feats, score, platform)
                st.session_state.result = {
                    "score": score, "breakdown": breakdown,
                    "feats": feats, "weaknesses": weaknesses,
                    "text": post_text, "platform": platform,
                }
                st.session_state.rewrite_result = None
        else:
            st.warning("Give me something to work with.")

   

    st.markdown('</div>', unsafe_allow_html=True)


# ╔═══════════════════════════╗
# ║  MIDDLE — Score           ║
# ╚═══════════════════════════╝
with col_mid:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Virality score</div>', unsafe_allow_html=True)

    if st.session_state.result:
        res   = st.session_state.result
        score = res["score"]

        if score >= 65:
            color_main  = T["success"]
            verdict_cls = "v-high"
            verdict_txt = "🚀 Strong potential"
        elif score >= 35:
            color_main  = T["warn"]
            verdict_cls = "v-medium"
            verdict_txt = "⚡ Needs a push"
        else:
            color_main  = T["danger"]
            verdict_cls = "v-low"
            verdict_txt = "📉 Won't travel far"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 54, "color": T["text"], "family": "Syne"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickfont": {"color": T["text_faint"], "size": 9},
                    "nticks": 6,
                },
                "bar": {"color": color_main, "thickness": 0.24},
                "bgcolor": T["border_soft"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": _rgba(T["danger"], 0.14)},
                    {"range": [35, 65], "color": _rgba(T["warn"], 0.14)},
                    {"range": [65,100], "color": _rgba(T["success"], 0.14)},
                ],
                "threshold": {
                    "line": {"color": color_main, "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=210,
            margin=dict(l=24, r=24, t=10, b=0),
            font={"family": "Syne"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="verdict-wrap">'
            f'<span class="verdict-badge {verdict_cls}">{verdict_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Feature breakdown
        st.markdown('<div class="feat-section-title">Signal breakdown</div>',
                    unsafe_allow_html=True)

        top     = res["breakdown"][:9]
        max_abs = max(abs(f["impact"]) for f in top) or 1

        for feat in top:
            imp      = feat["impact"]
            pct      = int(abs(imp) / max_abs * 100)
            fill_cls = "fill-pos" if imp >= 0 else "fill-neg"
            val_cls  = "val-pos"  if imp >= 0 else "val-neg"
            sign     = "+" if imp >= 0 else "−"
            st.markdown(f"""
            <div class="feat-item">
              <div class="feat-name">{feat['label']}</div>
              <div class="feat-track">
                <div class="feat-fill {fill_cls}" style="width:{pct}%"></div>
              </div>
              <div class="feat-val {val_cls}">{sign}{abs(imp):.2f}</div>
            </div>""", unsafe_allow_html=True)

        if res["weaknesses"]:
            st.markdown('<div class="weakness-wrap"><div class="weakness-title">Fix these first</div>', unsafe_allow_html=True)
            for w in res["weaknesses"]:
                st.markdown(f'<div class="weakness-item">{w}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-text">Score your post<br>to see results here.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ╔═══════════════════════════╗
# ║  RIGHT — Rewrite          ║
# ╚═══════════════════════════╝
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">AI rewrite</div>', unsafe_allow_html=True)

    env_api_key = os.getenv("GROQ_API_KEY", "")
    if env_api_key:
        api_key = env_api_key  # loaded silently from .env
    else:
        api_key = st.text_input(
            "GROQ API KEY",
            type="password",
            placeholder="gsk_...",
            help="Free key at console.groq.com",
            key="groq_api_key",
        )

    can_rewrite = (
        st.session_state.result is not None
        and len(api_key.strip()) > 10
    )

    btn_label = "✨  Rewrite to boost score" if st.session_state.result else "Score a post first →"
    if st.button(btn_label, disabled=not can_rewrite, key="rewrite_btn"):
        res = st.session_state.result
        with st.spinner("Llama is thinking..."):
            try:
                rewritten, explanation = rewrite_post(
                    original_text=res["text"],
                    platform=res["platform"],
                    score=res["score"],
                    weaknesses=res["weaknesses"],
                    api_key=api_key,
                )
                new_score, new_bd, new_feats = predict(
                    rewritten, res["platform"],
                    st.session_state.model,
                    st.session_state.scaler,
                )
                st.session_state.rewrite_result = {
                    "text": rewritten,
                    "explanation": explanation,
                    "new_score": new_score,
                    "old_score": res["score"],
                }
            except Exception as e:
                st.error(f"Rewrite failed: {e}")

    if not api_key:
        st.markdown(
            f'<div style="font-size:11px;color:{T["text_dim"]};margin-top:8px;">Add your Groq key above or set GROQ_API_KEY in .env — this also powers the chat assistant in the corner.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    if st.session_state.rewrite_result:
        rr    = st.session_state.rewrite_result
        old_s = rr["old_score"]
        new_s = rr["new_score"]
        delta = new_s - old_s
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        delta_cls = "rv-delta-pos" if delta > 0 else ("rv-delta-neg" if delta < 0 else "rv-delta-neu")

        st.markdown(f"""
        <div class="rewrite-scores">
          <div class="rscore-box rscore-before">
            <div class="rscore-label">Before</div>
            <div class="rscore-val rv-before">{old_s}</div>
          </div>
          <div class="rscore-box rscore-after">
            <div class="rscore-label">After</div>
            <div class="rscore-val rv-after">{new_s}</div>
          </div>
          <div class="rscore-box rscore-delta">
            <div class="rscore-label">Delta</div>
            <div class="rscore-val {delta_cls}">{delta_str}</div>
          </div>
        </div>
        <div class="rewrite-text">{rr['text']}</div>
        <div class="rewrite-explanation">💡 {rr['explanation']}</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        with st.expander("📋 Copy rewrite"):
            st.code(rr["text"], language=None)

        if st.button("⚡  Score the rewrite", key="rescore_btn"):
            new_score2, new_bd2, new_feats2 = predict(
                rr["text"], st.session_state.result["platform"],
                st.session_state.model, st.session_state.scaler,
            )
            new_weak2 = get_weaknesses(
                new_feats2, new_score2,
                st.session_state.result["platform"],
            )
            st.session_state.result.update({
                "score": new_score2, "breakdown": new_bd2,
                "feats": new_feats2, "weaknesses": new_weak2,
                "text": rr["text"],
            })
            st.session_state.rewrite_result = None
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">✨</div>
          <div class="empty-text">Score your post first,<br>then hit rewrite.<br>The AI will fix<br>exactly what's weak.<br><br>Or hit the 💬 in the corner<br>to chat it through instead.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ╔═══════════════════════════════════════════╗
# ║  FLOATING REWRITE CHATBOT (fixed corner)  ║
# ╚═══════════════════════════════════════════╝
with st.container(key="fab_container"):
    fab_icon = "✕" if st.session_state.chat_open else "💬"
    if st.button(fab_icon, key="fab_toggle_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

if st.session_state.chat_open:
    with st.container(key="chat_panel"):
        st.markdown('<div class="chat-header">✨ Rewrite Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chat-sub">Chats about your currently scored post — ask for a tone change, a shorter version, a platform switch, anything.</div>',
            unsafe_allow_html=True,
        )

        with st.container(key="chat_scroll", height=300):
            if not st.session_state.chat_messages:
                st.markdown(
                    '<div class="chat-empty">No messages yet. Try:<br>"make this funnier"<br>"rewrite it for Reddit"<br>"cut it to one line"</div>',
                    unsafe_allow_html=True,
                )
            for msg in st.session_state.chat_messages:
                role_cls = "chat-user" if msg["role"] == "user" else "chat-bot"
                st.markdown(f'<div class="chat-bubble {role_cls}">{msg["content"]}</div>', unsafe_allow_html=True)

        chat_api_key = os.getenv("GROQ_API_KEY", "") or st.session_state.get("groq_api_key", "") or ""
        user_msg = st.chat_input("Ask for a rewrite...", key="floating_chat_input")

        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            if len(chat_api_key.strip()) < 10:
                reply = "I need a Groq API key first — add it in the AI Rewrite panel, then come back and ask me anything."
            else:
                try:
                    reply = chat_with_rewriter(
                        messages=st.session_state.chat_messages,
                        context=st.session_state.result,
                        api_key=chat_api_key,
                    )
                except Exception as e:
                    reply = f"Something broke on my end: {e}"
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()