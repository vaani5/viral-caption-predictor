"""
Viral Post Predictor — Streamlit App
Light, readable, professional + quirky UI
"""

import streamlit as st
import plotly.graph_objects as go
import os, sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import load_model, predict, get_weaknesses, extract_features
from rewriter import rewrite_post

st.set_page_config(
    page_title="Viral or Spiral?",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* Force light background on every Streamlit container */
html, body { background: #F5F2EE !important; }
.stApp, .stApp > div, [class*="css"],
section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"],
div[data-testid="stHeader"],
div[data-testid="block-container"],
div.main, div.main > div {
  background: #F5F2EE !important;
  color: #1A1815 !important;
  font-family: 'Inter', sans-serif !important;
}

/* Kill ghost column boxes Streamlit renders */
div[data-testid="column"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="column"] > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* Kill any stray white boxes from stVerticalBlock */
div[data-testid="stVerticalBlock"] > div[style*="background"] {
  background: transparent !important;
}
/* Remove Streamlit's default element containers */
div.element-container {
  background: transparent !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1280px !important; }

/* HERO */
.hero {
  padding: 3rem 0 2rem;
  border-bottom: 1.5px solid #E0DAD2;
  margin-bottom: 2rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #B07D3A;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 4.5vw, 3.8rem);
  font-weight: 800;
  line-height: 1.0;
  color: #1A1815;
  margin-bottom: 0.55rem;
}
.hero-title .accent { color: #C8873A; }
.hero-sub {
  font-size: 14px;
  color: #7A7268;
  line-height: 1.65;
  max-width: 420px;
}
.hero-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #B0A898;
  text-align: right;
  line-height: 2;
}

/* PANEL */
.panel {
  background: #FFFFFF;
  border: 1.5px solid #E0DAD2;
  border-radius: 18px;
  padding: 1.5rem;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.panel-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #B0A898;
  margin-bottom: 1.1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #F0EBE4;
}

/* TEXTAREA */
.stTextArea textarea {
  background: #FAFAF8 !important;
  border: 1.5px solid #E0DAD2 !important;
  border-radius: 10px !important;
  color: #1A1815 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
}
.stTextArea textarea:focus {
  border-color: #C8873A !important;
  box-shadow: 0 0 0 3px rgba(200,135,58,0.1) !important;
}
.stTextArea label { display: none !important; }

/* SELECTBOX */
.stSelectbox > div > div {
  background: #FAFAF8 !important;
  border: 1.5px solid #E0DAD2 !important;
  border-radius: 10px !important;
  color: #1A1815 !important;
  font-size: 13px !important;
}
.stSelectbox label { display: none !important; }

/* BUTTONS */
.stButton > button {
  width: 100% !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 0.65rem 1.5rem !important;
  border: none !important;
  transition: all 0.18s ease !important;
}
.stButton > button:not(:disabled) {
  background: #C8873A !important;
  color: #FFFFFF !important;
}
.stButton > button:not(:disabled):hover {
  background: #B07530 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(200,135,58,0.25) !important;
}
.stButton > button:disabled {
  background: #F0EBE4 !important;
  color: #C0B8B0 !important;
  cursor: not-allowed !important;
}

/* TEXT INPUT */
.stTextInput input {
  background: #FAFAF8 !important;
  border: 1.5px solid #E0DAD2 !important;
  border-radius: 10px !important;
  color: #1A1815 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
}
.stTextInput input:focus { border-color: #C8873A !important; }
.stTextInput label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: #B0A898 !important;
}

/* CHAR COUNT */
.char-count {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  text-align: right;
  margin-top: -8px;
  margin-bottom: 10px;
  color: #B0A898;
}

/* VERDICT */
.verdict-wrap { text-align: center; margin: 0.5rem 0 1rem; }
.verdict-badge {
  display: inline-block;
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  padding: 5px 16px;
  border-radius: 99px;
  letter-spacing: 0.04em;
}
.v-low    { background: #FEF0EC; color: #C0441C; border: 1.5px solid #F5C5B5; }
.v-medium { background: #FEF8EC; color: #9A6B10; border: 1.5px solid #F0DDA0; }
.v-high   { background: #EDFAF3; color: #1A7A46; border: 1.5px solid #A8E0C0; }

/* FEATURE BARS */
.feat-section-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #B0A898;
  margin: 1.25rem 0 0.75rem;
  padding-top: 1rem;
  border-top: 1px solid #F0EBE4;
}
.feat-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.feat-name {
  font-size: 12px;
  color: #5A5550;
  width: 145px;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.feat-track {
  flex: 1;
  height: 5px;
  background: #F0EBE4;
  border-radius: 99px;
  overflow: hidden;
}
.feat-fill { height: 5px; border-radius: 99px; }
.fill-pos { background: linear-gradient(90deg, #2ECC80, #26A8D8); }
.fill-neg { background: linear-gradient(90deg, #E8604A, #E8904A); }
.feat-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  width: 38px;
  text-align: right;
  flex-shrink: 0;
}
.val-pos { color: #1A7A46; }
.val-neg { color: #C0441C; }

/* WEAKNESS */
.weakness-wrap {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #F0EBE4;
}
.weakness-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #B0A898;
  margin-bottom: 0.6rem;
}
.weakness-item {
  font-size: 12px;
  color: #4A4540;
  padding: 7px 10px 7px 12px;
  border-left: 2.5px solid #C8873A;
  margin-bottom: 6px;
  line-height: 1.55;
  background: #FEF8F4;
  border-radius: 0 8px 8px 0;
}

/* EMPTY STATE */
.empty-state {
  text-align: center;
  padding: 3.5rem 1rem;
}
.empty-icon { font-size: 2.2rem; margin-bottom: 0.75rem; }
.empty-text { font-size: 13px; color: #C0B8B0; line-height: 1.7; }

/* REWRITE */
.rewrite-scores {
  display: flex;
  gap: 8px;
  margin-bottom: 1rem;
}
.rscore-box {
  flex: 1;
  text-align: center;
  padding: 10px 8px;
  border-radius: 10px;
}
.rscore-before { background: #FEF0EC; border: 1.5px solid #F5C5B5; }
.rscore-after  { background: #EDFAF3; border: 1.5px solid #A8E0C0; }
.rscore-delta  { background: #F5F2EE; border: 1.5px solid #E0DAD2; }
.rscore-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #B0A898;
  margin-bottom: 4px;
}
.rscore-val {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 800;
}
.rv-before { color: #C0441C; }
.rv-after  { color: #1A7A46; }
.rv-delta-pos { color: #1A7A46; }
.rv-delta-neg { color: #C0441C; }
.rv-delta-neu { color: #9A8878; }

.rewrite-text {
  background: #F0FDF6;
  border: 1.5px solid #A8E0C0;
  border-radius: 12px;
  padding: 1.1rem 1.25rem;
  font-size: 14px;
  color: #1A3A2A;
  line-height: 1.75;
  margin-bottom: 0.75rem;
}
.rewrite-explanation {
  font-size: 11px;
  color: #1A7A46;
  font-style: italic;
  padding-left: 4px;
  line-height: 1.5;
}

/* PLATFORM HINT */
.platform-hint {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #B0A898;
  margin-top: 6px;
  margin-bottom: 8px;
  padding-left: 2px;
}

/* TIPS BOX */
.tips-box {
  background: #FAFAF8;
  border: 1.5px solid #E0DAD2;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-top: 1rem;
}
.tips-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #B0A898;
  margin-bottom: 0.5rem;
}
.tips-body {
  font-size: 12px;
  color: #8A8278;
  line-height: 2;
}

/* API KEY LOADED */
.api-loaded {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #1A7A46;
  margin-bottom: 0.75rem;
  background: #EDFAF3;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid #A8E0C0;
}

/* EXPANDER */
.streamlit-expanderHeader {
  background: #FAFAF8 !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  color: #7A7268 !important;
  border: 1px solid #E0DAD2 !important;
}

hr { border-color: #E0DAD2 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    with st.spinner("Warming up the algorithm..."):
        st.session_state.model, st.session_state.scaler = load_model()

if "result"         not in st.session_state: st.session_state.result = None
if "rewrite_result" not in st.session_state: st.session_state.rewrite_result = None


# ── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-eyebrow">⚡ Virality Intelligence · ML + GenAI</div>
    <div class="hero-title"><span class="accent">Viral or Spiral?</span></div>
    <div class="hero-sub">Paste your post. Get an ML virality score, a signal-by-signal breakdown, and an AI rewrite that fixes exactly what's weak.</div>
  </div>
  <div class="hero-meta">
    Model &nbsp;·&nbsp; XGBoost + SHAP<br>
    Rewriter &nbsp;·&nbsp; Llama 3.3 70B<br>
    Platforms &nbsp;·&nbsp; Twitter &nbsp;Reddit &nbsp;Instagram
  </div>
</div>
""", unsafe_allow_html=True)


# ── Platform config ─────────────────────────────────────────────────────────
PLATFORM_HINTS = {
    "twitter":   "60–220 chars · 1–2 hashtags · hooks + questions win",
    "reddit":    "Punchy title under 120 chars · no hashtags · no emojis",
    "instagram": "80–300 chars · 5–10 hashtags · CTA + emojis rewarded",
}
PLATFORM_ICONS = {"twitter": "🐦", "reddit": "🤖", "instagram": "📸"}
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
    cc_color = "#C0441C" if char_len > limit else "#B0A898"
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

    st.markdown("""
    <div class="tips-box">
      <div class="tips-title">What we check</div>
      <div class="tips-body">
        Hook phrases &nbsp;·&nbsp; Emotional punch<br>
        Power words &nbsp;·&nbsp; Personal story (I/my)<br>
        Length sweet spot &nbsp;·&nbsp; Readability<br>
        Hashtag strategy &nbsp;·&nbsp; Call to action
      </div>
    </div>
    """, unsafe_allow_html=True)

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
            color_main  = "#2ECC80"
            verdict_cls = "v-high"
            verdict_txt = "🚀 Strong potential"
        elif score >= 35:
            color_main  = "#E8C040"
            verdict_cls = "v-medium"
            verdict_txt = "⚡ Needs a push"
        else:
            color_main  = "#E8604A"
            verdict_cls = "v-low"
            verdict_txt = "📉 Won't travel far"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 54, "color": "#1A1815", "family": "Syne"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickfont": {"color": "#C0B8B0", "size": 9},
                    "nticks": 6,
                },
                "bar": {"color": color_main, "thickness": 0.24},
                "bgcolor": "#F0EBE4",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "#FDE8E4"},
                    {"range": [35, 65], "color": "#FEF8E4"},
                    {"range": [65,100], "color": "#E8F8EE"},
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

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "GROQ API KEY",
            type="password",
            placeholder="gsk_...",
            help="Free key at console.groq.com",
        )
    else:
        pass  # key loaded silently from .env

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
            '<div style="font-size:11px;color:#B0A898;margin-top:8px;">Add your Groq key above or set GROQ_API_KEY in .env</div>',
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
          <div class="empty-text">Score your post first,<br>then hit rewrite.<br>The AI will fix<br>exactly what's weak.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)