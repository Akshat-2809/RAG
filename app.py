import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eternal · PDF Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS + Responsive ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════
   FONTS
════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ════════════════════════════════════════
   CSS VARIABLES
════════════════════════════════════════ */
:root {
    --bg:           #050508;
    --purple:       #6c2bd9;
    --purple-soft:  rgba(108,43,217,0.12);
    --purple-glow:  rgba(108,43,217,0.35);
    --teal:         #22c5d3;
    --text:         #e8e4f0;
    --text-muted:   rgba(200,190,220,0.65);
    --text-dim:     rgba(200,190,220,0.35);
    --accent:       rgba(180,140,255,0.85);
    --glass-bg:     rgba(255,255,255,0.032);
    --glass-border: rgba(255,255,255,0.07);
    --radius-lg:    20px;
    --radius-md:    14px;
    --radius-sm:    10px;
    --card-pad:     2.2rem 2.4rem;
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
}

/* ════════════════════════════════════════
   RESET & BASE
════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 16px; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(108,43,217,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 110%, rgba(34,197,211,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(255,100,130,0.04) 0%, transparent 60%),
        var(--bg) !important;
    min-height: 100vh;
}

/* ════════════════════════════════════════
   HIDE STREAMLIT CHROME
════════════════════════════════════════ */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"] { display: none !important; }

/* ════════════════════════════════════════
   MAIN CONTAINER — responsive
════════════════════════════════════════ */
.main .block-container {
    max-width: 860px !important;
    width: 100% !important;
    padding: 0 1.5rem 5rem !important;
    margin: 0 auto !important;
}

/* ════════════════════════════════════════
   STAR CANVAS
════════════════════════════════════════ */
#star-canvas {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 0;
    opacity: 0.55;
}

/* ════════════════════════════════════════
   AMBIENT ORBS
════════════════════════════════════════ */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: orb-drift 18s ease-in-out infinite alternate;
}
.orb-1 { width:500px;height:500px;top:-180px;left:-120px;background:rgba(108,43,217,0.12);animation-delay:0s; }
.orb-2 { width:380px;height:380px;bottom:-100px;right:-80px;background:rgba(34,197,211,0.10);animation-delay:-6s; }
.orb-3 { width:260px;height:260px;top:40%;left:60%;background:rgba(255,100,130,0.07);animation-delay:-12s; }

/* ════════════════════════════════════════
   HERO
════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 5rem 1rem 3.5rem;
    position: relative;
    z-index: 1;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.4rem;
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--purple-soft);
    border: 1px solid rgba(108,43,217,0.3);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.8rem;
    animation: fadeSlideDown 0.8s ease both;
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(2.4rem, 8vw, 4.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #c8b4ff 40%, #22c5d3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeSlideDown 0.9s 0.1s ease both;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: clamp(0.85rem, 2.5vw, 1rem);
    font-weight: 300;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    animation: fadeSlideDown 1s 0.2s ease both;
    padding: 0 0.5rem;
}

/* ════════════════════════════════════════
   GLASS CARD
════════════════════════════════════════ */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: var(--card-pad);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeSlideUp 0.7s 0.3s ease both;
    z-index: 1;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,140,255,0.5), rgba(34,197,211,0.4), transparent);
}
.glass-card:hover {
    border-color: rgba(180,140,255,0.18);
    box-shadow: 0 0 40px rgba(108,43,217,0.10), 0 0 80px rgba(34,197,211,0.05);
}

/* ════════════════════════════════════════
   SECTION LABEL
════════════════════════════════════════ */
.section-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(180,140,255,0.7);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(180,140,255,0.15);
}

/* ════════════════════════════════════════
   STATUS CHIP
════════════════════════════════════════ */
.status-chip {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.45rem;
    font-family: var(--font-body);
    font-size: 0.75rem;
    font-weight: 500;
    color: rgba(134,239,172,0.9);
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    margin-bottom: 1.2rem;
    max-width: 100%;
    word-break: break-all;
}
.status-chip .dot {
    width: 6px; height: 6px; flex-shrink: 0;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse-dot 2s infinite;
}

/* ════════════════════════════════════════
   FILE UPLOADER
════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: rgba(108,43,217,0.05) !important;
    border: 2px dashed rgba(180,140,255,0.2) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.2rem !important;
    transition: border-color 0.3s, background 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(180,140,255,0.45) !important;
    background: rgba(108,43,217,0.09) !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: rgba(200,190,220,0.5) !important;
    font-family: var(--font-body) !important;
}
.stFileUploader label { color: rgba(200,190,220,0.5) !important; }

/* Ensure uploader action button text is always visible */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: rgba(230,225,245,0.95) !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus {
    background: rgba(255,255,255,0.18) !important;
    border-color: rgba(255,255,255,0.35) !important;
}

/* ════════════════════════════════════════
   BUTTON — full-width, touch-friendly
════════════════════════════════════════ */
.stButton { width: 100%; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6c2bd9 0%, #3b82f6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-display) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1.5rem !important;
    min-height: 48px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px var(--purple-glow), 0 4px 15px rgba(0,0,0,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(108,43,217,0.55), 0 8px 25px rgba(0,0,0,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ════════════════════════════════════════
   TEXT INPUT — font-size ≥ 16px stops iOS zoom
════════════════════════════════════════ */
.stTextInput > div > div,
[data-testid="stTextInputRootElement"] > div,
div[data-baseweb="input"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    padding: 0.7rem 1rem !important;
    min-height: 48px !important;
}

/* BaseWeb adds an inner wrapper that can create a second border effect */
div[data-baseweb="input"] > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Keep one clean outer border and center content vertically */
div[data-baseweb="input"] {
    display: flex !important;
    align-items: center !important;
    padding: 0.7rem 1rem !important;
    min-height: 52px !important;
}

.stTextInput > div > div:focus-within,
[data-testid="stTextInputRootElement"] > div:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: rgba(180,140,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(108,43,217,0.15) !important;
}
.stTextInput input,
[data-testid="stTextInputRootElement"] input,
div[data-baseweb="input"] input {
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    background: transparent !important;
    font-size: 1rem !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--text) !important;
    opacity: 1 !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1.35 !important;
}
.stTextInput input::placeholder,
[data-testid="stTextInputRootElement"] input::placeholder,
div[data-baseweb="input"] input::placeholder {
    color: var(--text-dim) !important;
    -webkit-text-fill-color: var(--text-dim) !important;
}
.stTextInput label {
    color: rgba(200,190,220,0.6) !important;
    font-family: var(--font-body) !important;
}

/* Streamlit DOM-specific selectors for reliable input visibility */
[data-testid="stTextInput"] input,
[data-testid="stTextInputRootElement"] input,
div[data-baseweb="input"] input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    opacity: 1 !important;
    caret-color: var(--text) !important;
}

/* Make the "Press Enter..." helper readable */
[data-testid="InputInstructions"],
[data-testid="InputInstructions"] * {
    color: rgba(232,228,240,0.72) !important;
    -webkit-text-fill-color: rgba(232,228,240,0.72) !important;
    opacity: 1 !important;
}

/* Prevent browser autofill from making text unreadable on bright background */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus,
textarea:-webkit-autofill,
textarea:-webkit-autofill:hover,
textarea:-webkit-autofill:focus {
    -webkit-text-fill-color: var(--text) !important;
    box-shadow: 0 0 0 1000px rgba(10, 10, 18, 0.92) inset !important;
    transition: background-color 5000s ease-in-out 0s !important;
    caret-color: var(--text) !important;
}

/* ════════════════════════════════════════
   ANSWER CARD
════════════════════════════════════════ */
.answer-card {
    background: rgba(108,43,217,0.07);
    border: 1px solid rgba(180,140,255,0.15);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
    z-index: 1;
}
.answer-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,140,255,0.6), transparent);
}
.answer-card-label {
    font-family: var(--font-display);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: rgba(180,140,255,0.7);
    margin-bottom: 1rem;
}
.answer-card-text {
    font-family: var(--font-body);
    font-size: clamp(0.88rem, 2.2vw, 0.97rem);
    font-weight: 300;
    line-height: 1.8;
    color: rgba(230,225,245,0.9);
    word-break: break-word;
}

/* ════════════════════════════════════════
   EXPANDER
════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    color: rgba(200,190,220,0.7) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 1rem !important;
    min-height: 48px !important;
}
.streamlit-expanderContent {
    background: transparent !important;
    border: none !important;
    padding-top: 0.5rem !important;
}

/* ════════════════════════════════════════
   SUCCESS / SPINNER
════════════════════════════════════════ */
.stSuccess, [data-testid="stNotification"] {
    background: rgba(34,197,94,0.08) !important;
    border: 1px solid rgba(34,197,94,0.2) !important;
    border-radius: 10px !important;
    color: rgba(134,239,172,0.9) !important;
    font-family: var(--font-body) !important;
}
.stSpinner > div { border-top-color: #b48cff !important; }

/* ════════════════════════════════════════
   SCROLLBAR
════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(180,140,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(180,140,255,0.4); }

/* ════════════════════════════════════════
   MISC
════════════════════════════════════════ */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
.stMarkdown p {
    color: rgba(200,190,220,0.75) !important;
    font-family: var(--font-body) !important;
    line-height: 1.7 !important;
}

/* ════════════════════════════════════════
   KEYFRAMES
════════════════════════════════════════ */
@keyframes fadeSlideDown {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(0.8); }
}
@keyframes orb-drift {
    from { transform:translate(0,0) scale(1); }
    to   { transform:translate(40px,30px) scale(1.08); }
}

/* ════════════════════════════════════════
   ██  RESPONSIVE BREAKPOINTS  ██
════════════════════════════════════════ */

/* ── Tablet  481–768 px ── */
@media (max-width: 768px) {
    :root { --card-pad: 1.5rem 1.6rem; --radius-lg: 16px; }

    .hero { padding: 3.5rem 0.5rem 2.5rem; }
    .hero-badge { font-size: 0.62rem; letter-spacing: 0.13em; margin-bottom: 1.4rem; }

    .main .block-container { padding: 0 1rem 4rem !important; }

    .orb-1 { width:300px; height:300px; top:-100px; left:-80px; }
    .orb-2 { width:220px; height:220px; bottom:-60px; right:-50px; }
    .orb-3 { display:none; }

    .answer-card { padding:1.3rem 1.4rem; }
}

/* ── Mobile  ≤ 480 px ── */
@media (max-width: 480px) {
    :root { --card-pad: 1.1rem 1.1rem; --radius-lg: 14px; --radius-md: 10px; }

    .main .block-container { padding: 0 0.6rem 3.5rem !important; }

    .hero { padding: 2.8rem 0.25rem 2rem; }
    .hero-badge { font-size: 0.58rem; letter-spacing: 0.1em; padding: 0.28rem 0.7rem; margin-bottom: 1.1rem; }
    .hero-sub   { font-size: 0.83rem; }

    .glass-card { margin-bottom: 0.9rem; }

    .status-chip { font-size: 0.7rem; }

    /* bigger tap targets */
    .stButton > button {
        font-size: 0.82rem !important;
        min-height: 52px !important;
        letter-spacing: 0.05em !important;
    }
    .stTextInput > div > div {
        font-size: 1rem !important;   /* prevents iOS zoom */
        min-height: 52px !important;
        padding: 0.75rem 0.9rem !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] {
        min-height: 52px !important;
        padding: 0.75rem 0.9rem !important;
        border-radius: 10px !important;
    }
    .stTextInput input { font-size: 1rem !important; }

    .answer-card { padding: 1rem 1.1rem; border-radius: 12px; }
    .answer-card-text { font-size: 0.9rem; }

    .orb-1 { width:200px; height:200px; filter:blur(55px); }
    .orb-2 { width:160px; height:160px; filter:blur(50px); }
    .orb-3 { display:none; }
}

/* ── Very small / landscape phones  ≤ 360 px ── */
@media (max-width: 360px) {
    :root { --card-pad: 0.9rem 0.9rem; }
    .hero { padding: 2rem 0.2rem 1.5rem; }
    .hero-badge { display: none; }
    .hero-sub   { font-size: 0.78rem; }
}

/* ── Desktop  ≥ 1024 px ── */
@media (min-width: 1024px) {
    .main .block-container { padding: 0 2rem 6rem !important; }
}
@media (min-width: 1280px) {
    .hero { padding: 6rem 0 4rem; }
}

/* ════════════════════════════════════════
   TOUCH / COARSE POINTER DEVICES
════════════════════════════════════════ */
@media (hover: none) and (pointer: coarse) {
    .stButton > button:hover { transform: none !important; }
    .glass-card:hover        { box-shadow: none; border-color: var(--glass-border); }
    .stButton > button  { min-height: 54px !important; }
    .stTextInput > div > div { min-height: 54px !important; }
}
</style>

<!-- Ambient orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<!-- Star canvas -->
<canvas id="star-canvas"></canvas>

<script>
(function () {
    const canvas = document.getElementById('star-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H, stars = [];

    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }

    /* Scale star count to screen size for performance */
    function starCount() { return W < 480 ? 70 : W < 768 ? 130 : 200; }

    function createStars() {
        stars = [];
        for (let i = 0; i < starCount(); i++) {
            stars.push({
                x: Math.random() * W,
                y: Math.random() * H,
                r: Math.random() * 1.1 + 0.2,
                alpha: Math.random() * 0.55 + 0.1,
                speed: Math.random() * 0.3 + 0.05,
                twinkle: Math.random() * Math.PI * 2
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        stars.forEach(s => {
            s.twinkle += s.speed * 0.04;
            const a = s.alpha * (0.5 + 0.5 * Math.sin(s.twinkle));
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(220,210,255,${a})`;
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }

    resize(); createStars(); draw();
    window.addEventListener('resize', () => { resize(); createStars(); });
})();
</script>

<!-- Hero -->
<div class="hero">
    <div class="hero-badge">✦ AI-Powered &nbsp;·&nbsp; RAG Engine &nbsp;·&nbsp; Mistral AI</div>
    <div class="hero-title">Eternal PDF</div>
    <div class="hero-sub">Drop any PDF. Ask anything. Get precise, document-grounded answers in seconds.</div>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="section-label">✦ &nbsp;Upload Document</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type="pdf",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ── Process PDF ───────────────────────────────────────────────────────────────
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    file_size_kb = os.path.getsize(file_path) / 1024
    display_name = (
        uploaded_file.name if len(uploaded_file.name) <= 28
        else uploaded_file.name[:25] + "…"
    )

    st.markdown(f"""
    <div class="glass-card" style="animation-delay:0.1s;">
        <div class="status-chip">
            <span class="dot"></span>
            {display_name} &nbsp;·&nbsp; {file_size_kb:.1f} KB
        </div>
        <div class="section-label">✦ &nbsp;Index Document</div>
    """, unsafe_allow_html=True)

    if st.button("⚡  Build Knowledge Index"):
        with st.spinner("Embedding document chunks into vector space…"):
            # Delete old DB so previous PDF data doesn't mix with the new one
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            loader = PyPDFLoader(file_path)
            docs   = loader.load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)
            embeds = MistralAIEmbeddings(model="mistral-embed")
            vs     = Chroma.from_documents(
                documents=chunks, embedding=embeds, persist_directory="chroma_db"
            )
            vs.persist()
        st.success(f"✓ Indexed {len(chunks)} chunks across {len(docs)} pages")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Q&A Section ───────────────────────────────────────────────────────────────
if os.path.exists("chroma_db"):
    embeds      = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeds)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    llm    = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise, helpful AI assistant.\n"
         "Use ONLY the provided context to answer the question.\n"
         "Structure your answer clearly. If the answer is not present "
         "in the context, say: \"I could not find the answer in the document.\""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    st.markdown("""
    <div class="glass-card" style="animation-delay:0.2s;">
        <div class="section-label">✦ &nbsp;Ask Anything</div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        label="",
        placeholder="e.g. What are the key findings in chapter 3?",
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if query:
        with st.spinner("Searching document & generating answer…"):
            retrieved_docs = retriever.invoke(query)
            context        = "\n\n".join([d.page_content for d in retrieved_docs])
            final_prompt   = prompt.invoke({"context": context, "question": query})
            response       = llm.invoke(final_prompt)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-card-label">✦ &nbsp;Answer</div>
            <div class="answer-card-text">{response.content}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍  View Retrieved Source Passages"):
            for i, doc in enumerate(retrieved_docs, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border-left: 2px solid rgba(180,140,255,0.35);
                    border-radius: 0 8px 8px 0;
                    padding: 0.8rem 1rem;
                    margin-bottom: 0.75rem;
                    font-family: 'DM Sans', sans-serif;
                    font-size: 0.82rem;
                    color: rgba(200,190,220,0.7);
                    line-height: 1.65;
                    word-break: break-word;
                ">
                    <span style="color:rgba(180,140,255,0.6);font-size:0.62rem;
                                 letter-spacing:0.15em;text-transform:uppercase;">
                        Passage {i} &nbsp;·&nbsp; Page {page}
                    </span><br><br>
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    color: rgba(200,190,220,0.2);
    letter-spacing: 0.08em;
    z-index: 1; position: relative;
">
    ETERNAL &nbsp;·&nbsp; Powered by Mistral AI &amp; LangChain &nbsp;·&nbsp; RAG Architecture
</div>
""", unsafe_allow_html=True)