# speech.py
# BOB AI Multilingual Product Assistant
# Record / type → Understand → Match products → Answer → Speak

import ast
import difflib
import hmac
import html
import io
import os
import re
import textwrap
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from google.genai import types
from gtts import gTTS

# =====================================================
# PAGE  (must be the first Streamlit call)
# =====================================================

st.set_page_config(
    page_title="BOB AI Multilingual Product Assistant",
    page_icon="💬",
    layout="wide",
)

# =====================================================
# CONSTANTS
# =====================================================

BANNER_PATH = r"img - Copy (2).png"
PRODUCT_DIR = "Bank_Product"

SAMPLE_RATE = 16000
MIN_AUDIO_BYTES = 5000  # recordings smaller than this are treated as silence

DEFAULT_VOICE_MODEL = "gemini-flash-lite-latest"
DEFAULT_GEN_MODEL = "gemini-flash-lite-latest"

DOMAINS = [
    "home loan",
    "vehicle loan",
    "education loan",
    "personal loan",
    "mortgage loan",
    "term deposits",
    "savings account",
    "salary account",
    "current account",
    "debit card",
]

LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "bengali": "bn",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "gujarati": "gu",
}

IGNORED_MODEL_KEYS = [
    "image", "live", "emb", "tts", "clip", "deep", "aqa", "veo", "lyria",
    "audio", "computer", "robo", "pro", "gemma", "2.0",
]

SAMPLE_QUESTIONS = [
    "I want a home loan to buy a flat. What are the options?",
    "I need a salary account with good benefits.",
    "मुझे गाड़ी खरीदने के लिए लोन चाहिए",
]

EXTRACT_KEYS = ["LANGUAGE", "TRANSCRIPT", "TRANSLATED_QUESTION", "DOMAIN", "KEYWORDS", "KEYWORDS_NATIVE"]
ANSWER_KEYS = ["RESPONSE", "TRANSLATE"]

# =====================================================
# STYLE  (BOB colours – unchanged, plus a few additions)
# =====================================================

st.markdown("""
<style>

/* ===== BOB PALETTE =====
   orange  #ff7a00 / #ff8c1a / #ffa64d     blue  #0059b3     navy  #002e6e
   tints   #fff4e6 (orange)  #eef5ff (blue)   lines #e3eaf6                 */

/* ===== GLOBAL ===== */
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #fffaf4 55%, #f2f7ff 100%);
    font-family: 'Segoe UI', sans-serif;
}

header[data-testid="stHeader"] { background: transparent; }

/* ===== HEADINGS ===== */
h1, h2, h3, h4 {
    color: #002e6e !important;
    font-weight: 700;
}

h2 {
    border-left: 5px solid #ff7a00;
    padding-left: 10px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffb870 0%, #ff8c1a 50%, #ff7a00 100%);
    border-right: 1px solid rgba(0, 46, 110, 0.15);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

section[data-testid="stSidebar"] h2 {
    border-left: none;
    padding-left: 0;
    border-bottom: 2px solid rgba(255, 255, 255, 0.45);
    padding-bottom: 4px;
    font-size: 1.15rem;
}

/* ===== BUTTONS ===== */
/* secondary = BOB blue */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0059b3, #003f80);
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 6px rgba(0, 63, 128, 0.25);
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #0a6bd0, #0059b3);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 63, 128, 0.3);
}

/* primary = BOB orange */
.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, #ff8c1a, #ff7a00);
    box-shadow: 0 2px 6px rgba(255, 122, 0, 0.35);
}

.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: linear-gradient(135deg, #ffa64d, #ff8c1a);
    box-shadow: 0 4px 10px rgba(255, 122, 0, 0.4);
}

/* button text must stay white (markdown rule below would turn it dark) */
.stButton button p,
.stDownloadButton button p,
div[data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important;
    font-weight: 600;
}

/* sidebar buttons: navy so they stand out on orange */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #003f80, #002e6e);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

/* ===== INFO / SUCCESS / WARNING / ERROR ===== */
div[data-testid="stAlert"] {
    border-radius: 10px;
    padding: 12px;
}

div[data-testid="stAlert"][kind="info"] {
    background-color: #eef5ff !important;
    border-left: 5px solid #0059b3 !important;
}

div[data-testid="stAlert"][kind="info"] * { color: #1f2a3c !important; }

div[data-testid="stAlert"][kind="success"] {
    background-color: #e6f6ec !important;
    border-left: 5px solid #28a745 !important;
}

div[data-testid="stAlert"][kind="success"] * { color: #1f2a3c !important; }

div[data-testid="stAlert"][kind="warning"] {
    background-color: #fff4e0 !important;
    border-left: 5px solid #ff8c1a !important;
}

div[data-testid="stAlert"][kind="warning"] * { color: #1f2a3c !important; }

div[data-testid="stAlert"][kind="error"] {
    background-color: #fdecee !important;
    border-left: 5px solid #dc3545 !important;
}

div[data-testid="stAlert"][kind="error"] * { color: #1f2a3c !important; }

/* ===== INPUTS ===== */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
}

div[role="listbox"] { color: black !important; }

div[data-baseweb="textarea"] {
    border-radius: 10px;
    border: 1px solid #c9d8ee;
    background: #ffffff;
}

div[data-baseweb="textarea"]:focus-within {
    border-color: #ff7a00 !important;
    box-shadow: 0 0 0 2px rgba(255, 122, 0, 0.2);
}

div[data-testid="stExpander"] {
    border: 1px solid #e3eaf6;
    border-radius: 12px;
    background: #ffffff;
}

/* ===== TABS ===== */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #002e6e;
    background-color: #eef5ff;
    border-radius: 8px 8px 0 0;
    margin-right: 4px;
    padding: 6px 12px;
    font-size: 13px;
    transition: all 0.2s ease-in-out;
}

button[data-baseweb="tab"]:hover {
    background-color: #dcebff;
    color: #0059b3;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ff8c1a, #ff7a00) !important;
    color: white !important;
    border-bottom: 2px solid #ff7a00;
}

div[data-baseweb="tab-list"] { border-bottom: 1px solid #e3eaf6; }

div[data-baseweb="tab-highlight"] { background-color: transparent !important; }

/* ===== MARKDOWN TEXT ===== */
p, li, span { color: black; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { height: 6px; width: 6px; }

::-webkit-scrollbar-thumb {
    background: #0059b3;
    border-radius: 10px;
}

[data-testid="stMarkdownContainer"] p {
    margin-top: 0rem !important;
    margin-bottom: 0.6rem !important;
    padding-left: 0rem !important;
    color: #1f2a3c !important;
    line-height: 1.6;
    font-size: 15px;
}

[data-testid="stMarkdownContainer"] li {
    color: #1f2a3c !important;
    margin-bottom: 0.25rem;
}

[data-testid="stMarkdownContainer"] strong {
    color: #ff7a00 !important;
    font-weight: 700;
}

[data-testid="stMarkdownContainer"] code {
    background-color: #fff4e6;
    color: #d35400;
    padding: 2px 5px;
    border-radius: 5px;
}

[data-testid="stMarkdownContainer"] a {
    color: #0059b3 !important;
    text-decoration: none;
    font-weight: 500;
}

/* ===== BOB SHADOW BOX ===== */
.bob-box {
    background: linear-gradient(135deg, #fff7ed, #fff1e0);
    border-left: 5px solid #ff7a00;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 46, 110, 0.08);
    margin-top: 10px;
    margin-bottom: 15px;
    color: #1f2a3c;
    line-height: 1.6;
    font-size: 15px;
}

/* blue variant for what the assistant understood */
.bob-box.blue {
    background: linear-gradient(135deg, #f3f8ff, #e8f1ff);
    border-left-color: #0059b3;
}

.bob-box p {
    margin: 0;
    line-height: 1.6;
    font-size: 15px;
}

.bob-box strong { color: #ff7a00; }

/* ===== DASHBOARD ADDITIONS ===== */

.kpi-card {
    background: linear-gradient(180deg, #ffffff, #fff8f0);
    border: 1px solid #ffe0bd;
    border-top: 4px solid #ff7a00;
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: 0 4px 14px rgba(255, 122, 0, 0.1);
    height: 100%;
}

.kpi-card.blue {
    background: linear-gradient(180deg, #ffffff, #f2f7ff);
    border: 1px solid #cfe0f7;
    border-top: 4px solid #0059b3;
    box-shadow: 0 4px 14px rgba(0, 89, 179, 0.1);
}

.kpi-label { color: #5b6b82; font-size: 13px; font-weight: 600; }
.kpi-value { color: #002e6e; font-size: 28px; font-weight: 700; line-height: 1.25; }
.kpi-sub   { color: #7a8699; font-size: 12px; min-height: 16px; }

.chip {
    display: inline-block;
    background: #e6f0ff;
    color: #003f80;
    border: 1px solid #b3d1ff;
    border-radius: 999px;
    padding: 3px 12px;
    margin: 2px 6px 2px 0;
    font-size: 13px;
    font-weight: 600;
}

.chip.orange {
    background: #fff1e0;
    color: #c25e00;
    border-color: #ffcf99;
}

.stage-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px 0; }

.stage {
    background: #ffffff;
    border: 1px solid #cfe0f7;
    border-left: 4px solid #0059b3;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 600;
    color: #002e6e;
}

.stage b { color: #ff7a00; }

.how-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 8px 0 18px 0;
}

.how-card {
    background: #ffffff;
    border: 1px solid #ffe0bd;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 3px 10px rgba(255, 122, 0, 0.08);
}

.how-card:nth-child(even) {
    border-color: #cfe0f7;
    box-shadow: 0 3px 10px rgba(0, 89, 179, 0.08);
}

.how-num {
    display: inline-block;
    background: #ff7a00;
    color: #ffffff;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    line-height: 26px;
    text-align: center;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
}

.how-card:nth-child(even) .how-num { background: #0059b3; }

.how-title { color: #002e6e; font-weight: 700; font-size: 15px; }
.how-text  { color: #4a5a70; font-size: 13px; line-height: 1.45; }

.fallback-banner {
    background: linear-gradient(90deg, #ff7a00, #ff8c1a);
    color: #ffffff;
    border-radius: 12px;
    padding: 18px 24px;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# BANNER
# =====================================================

if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, width="stretch")
else:
    st.markdown(
        '<div class="fallback-banner">BOB AI Multilingual Product Assistant</div>',
        unsafe_allow_html=True,
    )

# =====================================================
# HTML HELPERS
# =====================================================


def esc(text) -> str:
    """Escape model output before placing it inside HTML."""
    return html.escape(str(text or "")).replace("\n", "<br>")


def bob_box(text, tone="") -> str:
    """tone: '' (orange, answers) or 'blue' (what the assistant understood)."""
    return (
        f'<div class="bob-box {tone}" style="padding: 1rem; border-radius: 10px; '
        f'word-wrap: break-word;">{esc(text) or "&nbsp;"}</div>'
    )


def chips(items, css_class="") -> str:
    if not items:
        return '<span class="chip">—</span>'
    return "".join(f'<span class="chip {css_class}">{esc(i)}</span>' for i in items)


def kpi_card(label, value, sub="", css_class="") -> str:
    return (
        f'<div class="kpi-card {css_class}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div>'
        "</div>"
    )


# =====================================================
# API KEY  (Streamlit secrets → env var fallback)
# =====================================================


def get_api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = os.environ.get("GEMINI_API_KEY")
    return key or None


API_KEY = get_api_key()

if not API_KEY:
    st.error(
        "Gemini API key not found. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` "
        "(local) or to **App settings → Secrets** on Streamlit Cloud."
    )
    st.code('GEMINI_API_KEY = "your-gemini-api-key"', language="toml")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


client = get_client(API_KEY)

# =====================================================
# MODEL LIST  (cached – no API call on every rerun)
# =====================================================


@st.cache_data(ttl=3600, show_spinner="Loading models…")
def list_models(_client):
    voice_models, gen_models = [], []

    for m in _client.models.list():
        name = m.name.lower()
        clean = m.name.replace("models/", "")

        if not any(x in name for x in IGNORED_MODEL_KEYS):
            voice_models.append(clean)

        if "gemma" in name:
            gen_models.append(clean)

    return sorted(voice_models), sorted(gen_models)


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


try:
    voice_models, gemma_models = list_models(client)
except Exception as e:
    st.sidebar.warning(f"Could not load the model list ({e}). Using defaults.")
    voice_models, gemma_models = [DEFAULT_VOICE_MODEL], [DEFAULT_GEN_MODEL]

generation_models = voice_models + gemma_models

# =====================================================
# SESSION STATE
# =====================================================

for key, default in {
    "audio_data": None,
    "confirmed_audio": False,
    "pending_text": None,
    "current": None,
    "history": [],
    "dev_unlocked": False,
    "dev_open": False,
    "dev_files": {},
    "dev_selected": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =====================================================
# SESSION PERSISTENCE
# st.session_state is wiped by a browser refresh, a dropped connection or
# an idle reconnect. A random id in the URL (?sid=...) lets us keep a copy of
# the important values in server memory and restore it on the next run.
# =====================================================

PERSIST_KEYS = ["history", "current", "audio_data", "dev_files"]
MAX_HISTORY = 50


@st.cache_resource
def _server_store():
    return {}


def _session_id():
    sid = st.query_params.get("sid")
    if not sid:
        sid = uuid.uuid4().hex[:16]
        st.query_params["sid"] = sid
    return sid


SID = _session_id()

if "_restored" not in st.session_state:
    saved = _server_store().get(SID)
    if saved:
        for key in PERSIST_KEYS:
            if key in saved:
                st.session_state[key] = saved[key]
    st.session_state["_restored"] = True


def persist():
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    _server_store()[SID] = {k: st.session_state[k] for k in PERSIST_KEYS}


def rerun():
    """Save first, then rerun, so nothing set on this run is lost."""
    persist()
    st.rerun()


# =====================================================
# DEVELOPER OPTIONS - constants
# =====================================================


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# Optional: set DEV_PASSWORD in secrets to change it without touching the code.
DEV_PASSWORD = str(get_secret("DEV_PASSWORD", "acoe"))

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("## ⚙️ Model Selection")

MODEL = st.sidebar.selectbox(
    "🧠 Voice Model",
    voice_models,
    key="voice_model",
    index=default_index(voice_models, DEFAULT_VOICE_MODEL),
)

GEN_MODEL = st.sidebar.selectbox(
    "🧠 Generation Model",
    generation_models,
    key="generation_model",
    index=default_index(generation_models, DEFAULT_GEN_MODEL),
)

st.sidebar.markdown("## 🎛️ Preferences")

AUTOPLAY = st.sidebar.toggle("🔊 Auto-play spoken answer", value=True)

AUTO_SUBMIT = st.sidebar.toggle(
    "⚡ Auto-submit after recording",
    value=False,
    help="Skip the 'Get answer' click: the question is sent as soon as you stop speaking.",
)

TOP_N = st.sidebar.slider(
    "🏦 Products to consider", min_value=3, max_value=30, value=10,
    help="How many matching products are sent to the model to build the answer.",
)

st.sidebar.markdown("## 🧹 Session")

if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.session_state.current = None
    rerun()

st.sidebar.markdown("## 🛠️ Developer Options")

if not st.session_state.dev_unlocked:
    dev_pw = st.sidebar.text_input("🔒 Developer password", type="password", key="dev_pw")
    if dev_pw:
        if hmac.compare_digest(dev_pw.encode(), DEV_PASSWORD.encode()):
            st.session_state.dev_unlocked = True
            rerun()
        else:
            st.sidebar.error("Incorrect password.")
else:
    st.sidebar.success("Developer mode is on")

    if st.session_state.dev_open:
        if st.sidebar.button("⬅️ Back to Assistant", key="dev_back_sidebar"):
            st.session_state.dev_open = False
            rerun()
    elif st.sidebar.button("📥 Open Data Downloader", key="dev_open_btn"):
        st.session_state.dev_open = True
        rerun()

    if st.sidebar.button("🔒 Lock developer mode", key="dev_lock"):
        st.session_state.dev_unlocked = False
        st.session_state.dev_open = False
        rerun()

# =====================================================
# PIPELINE
# =====================================================


class PipelineError(Exception):
    """Raised with a user-friendly message."""


def build_extract_prompt(kind, text=None):
    if kind == "audio":
        steps = (
            "1. Detect the spoken language.\n"
            "2. Transcribe the speech accurately.\n"
            "3. Translate the speech into English."
        )
        source = ""
    else:
        steps = (
            "1. Detect the language of the customer text below.\n"
            "(Strictly Consider the original languages whether it may have or have not been typed in latin text, like for Hinglish - consider Hindi, for Banglish - consider Bengali and so on)\n"
            "2. Copy the customer text as the transcript, unchanged.\n"
            "3. Translate the text into English."
        )
        source = f"\nCustomer text:\n{text}\n"

    return f"""
You are a multilingual & financial AI assistant.
{source}
Tasks:

{steps}
4. Find the exact Domain Name from these domains: {DOMAINS}, return exactly single value
5. Find Only the Essential English Keywords as a Python List (can only be set of unigrams), ignore general keywords like parts-of-speech keywords (pronouns, adjective, verb, adverb, prepositions, conjunctions, interjections and so on) and also ignore general financial keywords that already present in domain name (like 'loan', 'atm', 'credit', 'debit', 'account', 'savings' etc.)
6. For every keyword from task 5, give the same keyword in the customer's original language and script, as a Python dict {{"english keyword": "native keyword"}}. If the language is English, repeat the English keyword.

Return ONLY in this exact format:

LANGUAGE:
<detected language>

TRANSCRIPT:
<original transcript>

TRANSLATED_QUESTION:
<english translation>

DOMAIN:
<domain name>

KEYWORDS:
<keywords list>

KEYWORDS_NATIVE:
<keywords dict>
"""


def build_answer_prompt(question, language, products_text):
    return textwrap.dedent(f"""
        You are a multilingual & financial AI assistant.

        Customer Requirement:
        {question}

        Detected User Language:
        {language}

        Database Results:
        {products_text}

        Tasks:

        1. Analyze the customer requirement carefully.
        2. Find the best possible financial/banking solution from the database.
        3. Explain WHY this solution is suitable.
        4. Respond professionally and remove any markdown things like '#', '*' or any other, just plain text would be converted from text-to-speech further. Return any URLs if required.
        5. Translate the final response back into {language}.

        Return ONLY in this exact format:

        RESPONSE:
        <best possible solution according to user requirement>

        TRANSLATE:
        <translated response>
    """)


def parse_sections(text, keys):
    """Split 'KEY: value' blocks. Tolerates **KEY:** and same-line values."""
    pattern = r"(?m)^[\s*#]*(" + "|".join(keys) + r")[\s*]*:[\s*]*"
    parts = re.split(pattern, text or "")
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def normalize_domain(raw):
    cleaned = re.sub(r"[^a-z ]", "", (raw or "").lower()).strip()
    if cleaned in DOMAINS:
        return cleaned
    for d in DOMAINS:
        if d in cleaned:
            return d
    close = difflib.get_close_matches(cleaned, DOMAINS, n=1, cutoff=0.6)
    return close[0] if close else None


def parse_keywords(raw):
    raw = (raw or "").strip()
    items = []
    try:
        value = ast.literal_eval(raw[raw.index("["): raw.rindex("]") + 1])
        if isinstance(value, (list, tuple, set)):
            items = [str(k) for k in value]
    except Exception:
        items = re.split(r"[,\n]", raw)

    cleaned = []
    for k in items:
        k = k.strip(" '\"[]`*").lower()
        if k and k not in cleaned:
            cleaned.append(k)
    return cleaned


def parse_keyword_map(raw, keywords):
    """english keyword -> native keyword; falls back to the English word."""
    raw = (raw or "").strip()
    mapping = {}
    try:
        value = ast.literal_eval(raw[raw.index("{"): raw.rindex("}") + 1])
        if isinstance(value, dict):
            mapping = {str(k).strip().lower(): str(v).strip() for k, v in value.items()}
    except Exception:
        pass
    return {k: mapping.get(k, k) for k in keywords}


@st.cache_data(show_spinner=False)
def load_products(domain):
    path = os.path.join(PRODUCT_DIR, f"{'_'.join(domain.split())}.xlsx")
    return pd.read_excel(path)


def retrieve_products(df, keywords, top_n):
    """Rank rows by how many keywords they contain (original: one ordered regex)."""
    # fillna + map(str): empty Excel cells are NaN floats and would break the join
    blob = df.fillna("").apply(lambda row: " ".join(map(str, row)).lower(), axis=1)
    hits = blob.apply(lambda text: sum(1 for k in keywords if k in text))
    matched = hits > 0

    if matched.any():
        out = (
            df.assign(**{"Keyword hits": hits})[matched]
            .sort_values("Keyword hits", ascending=False, kind="stable")
            .head(top_n)
        )
        return out, "matched"

    return df.assign(**{"Keyword hits": 0}).head(top_n), "fallback"


def stream_text(model, contents, placeholder, tone=""):
    text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents):
        if chunk.text:
            text += chunk.text
            placeholder.markdown(bob_box(text, tone), unsafe_allow_html=True)
    return text


def synthesize(text, language):
    code = LANGUAGE_MAP.get((language or "").strip().lower(), "en")
    if code == "mr":  # gTTS Marathi fallback (kept from original)
        code = "hi"
    buf = io.BytesIO()
    gTTS(text=re.sub(r"[#*`>]+", "", text), lang=code).write_to_fp(buf)
    return buf.getvalue()


def run_pipeline(job, status):
    timings = {}
    result = {
        "id": int(time.time() * 1000),
        "time": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "kind": job["kind"],
        "input_audio": job.get("audio"),
        "played": False,
    }

    # ---------- 1. understand ----------
    status.update(label="Understanding the question…")
    st.write("🎧 Understanding the question…" if job["kind"] == "audio" else "⌨️ Reading the question…")
    placeholder = st.empty()

    t0 = time.perf_counter()
    if job["kind"] == "audio":
        contents = [
            types.Part.from_bytes(data=job["audio"], mime_type="audio/wav"),
            build_extract_prompt("audio"),
        ]
    else:
        contents = build_extract_prompt("text", job["text"])

    raw_extract = stream_text(MODEL, contents, placeholder, "blue")
    timings["Understand"] = time.perf_counter() - t0

    sections = parse_sections(raw_extract, EXTRACT_KEYS)
    domain = normalize_domain(sections.get("DOMAIN"))
    question = sections.get("TRANSLATED_QUESTION", "").strip()

    if not question:
        raise PipelineError("Could not understand the question. Please try again and speak clearly.")
    if not domain:
        raise PipelineError(
            "This question doesn't match a supported product category. "
            "Try asking about: " + ", ".join(d.title() for d in DOMAINS) + "."
        )

    keywords = parse_keywords(sections.get("KEYWORDS"))
    result.update(
        raw_extract=raw_extract,
        language=sections.get("LANGUAGE", "").strip() or "Unknown",
        transcript=sections.get("TRANSCRIPT", "").strip(),
        translated_question=question,
        domain=domain,
        keywords=keywords,
        keyword_native=parse_keyword_map(sections.get("KEYWORDS_NATIVE"), keywords),
    )

    # ---------- 2. match products ----------
    status.update(label="Finding matching products…")
    st.write(f"🏦 Searching {domain.title()} products…")

    t0 = time.perf_counter()
    try:
        products = load_products(domain)
    except FileNotFoundError:
        raise PipelineError(
            f"Product data for '{domain.title()}' was not found. "
            f"Expected file: {PRODUCT_DIR}/{'_'.join(domain.split())}.xlsx"
        )

    matched, match_mode = retrieve_products(products, result["keywords"], TOP_N)
    timings["Match products"] = time.perf_counter() - t0
    result.update(matched=matched, match_mode=match_mode, catalogue_size=len(products))

    # ---------- 3. compose answer ----------
    status.update(label="Preparing your answer…")
    st.write("🤖 Preparing your answer…")
    placeholder = st.empty()

    products_text = matched.drop(columns="Keyword hits").to_string(index=False)
    prompt = build_answer_prompt(question, result["language"], products_text)

    t0 = time.perf_counter()
    raw_answer = stream_text(GEN_MODEL, prompt, placeholder)
    timings["Compose answer"] = time.perf_counter() - t0

    answer_sections = parse_sections(raw_answer, ANSWER_KEYS)
    answer_en = answer_sections.get("RESPONSE", "").strip() or raw_answer.strip()
    answer_local = answer_sections.get("TRANSLATE", "").strip() or answer_en
    result.update(raw_answer=raw_answer, answer_en=answer_en, answer_local=answer_local)

    # ---------- 4. speak ----------
    status.update(label="Generating voice…")
    st.write("🔊 Generating voice…")

    t0 = time.perf_counter()
    try:
        result["tts"] = synthesize(answer_local, result["language"])
        result["tts_error"] = None
    except Exception as e:  # voice is a bonus – never lose the text answer
        result["tts"] = None
        result["tts_error"] = str(e)
    timings["Speak"] = time.perf_counter() - t0

    result["timings"] = timings
    result["total"] = sum(timings.values())
    return result


# =====================================================
# RENDERING
# =====================================================


def render_kpis(slot):
    history = st.session_state.history
    n = len(history)
    languages = sorted({h["language"].title() for h in history if h["language"]})
    avg = sum(h["total"] for h in history) / n if n else None
    rated = [h["feedback"] for h in history if h["feedback"] is not None]
    helpful = f"{round(100 * sum(rated) / len(rated))}%" if rated else "—"

    with slot.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            kpi_card("💬 Questions answered", n, "this session"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            kpi_card("🌐 Languages used", len(languages), ", ".join(languages[:3]) or "—", "blue"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            kpi_card("⏱️ Avg. response time", f"{avg:.1f}s" if avg else "—", "question to spoken answer"),
            unsafe_allow_html=True,
        )
        c4.markdown(
            kpi_card("👍 Helpful answers", helpful, f"{len(rated)} rated" if rated else "rate an answer below", "blue"),
            unsafe_allow_html=True,
        )


def render_welcome():
    st.markdown("### 🌍 Ask about BOB products in your own language")

    steps = [
        ("Record or type", "Speak or type your question in any supported language."),
        ("Understand", "Speech is transcribed, translated and mapped to a product category."),
        ("Match", "Your keywords are matched against the product catalogue."),
        ("Answer", "The best-fit product is explained, and why it suits you."),
        ("Speak", "The answer is translated back and read out loud."),
    ]
    cards = "".join(
        f'<div class="how-card"><div class="how-num">{i}</div>'
        f'<div class="how-title">{t}</div><div class="how-text">{d}</div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    st.markdown(f'<div class="how-grid">{cards}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Languages**")
        st.markdown(chips([l.title() for l in LANGUAGE_MAP], "orange"), unsafe_allow_html=True)
    with right:
        st.markdown("**Product categories**")
        st.markdown(chips([d.title() for d in DOMAINS]), unsafe_allow_html=True)

    st.info("Tap the microphone above, or switch to **⌨️ Type**, to get started.")


def render_result(r):
    # --- at-a-glance summary ---
    st.markdown(
        chips([f"🌐 {r['language'].title()}", f"🏦 {r['domain'].title()}"], "orange")
        + chips(r["keywords"]),
        unsafe_allow_html=True,
    )

    stages = "".join(
        f'<div class="stage">✅ {name} <b>{secs:.1f}s</b></div>'
        for name, secs in r["timings"].items()
    )
    stages += f'<div class="stage">Total <b>{r["total"]:.1f}s</b></div>'
    st.markdown(f'<div class="stage-row">{stages}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧾 Raw Response",
        "📌 Extracted Information",
        "🏦 Matched Products",
        "🤖 Model Response",
        "🌐 Translated Response",
    ])

    with tab1:
        st.markdown("## RAW RESPONSE")
        st.markdown(bob_box(r["raw_extract"], "blue"), unsafe_allow_html=True)

    with tab2:
        if r.get("input_audio"):
            st.markdown("## Your Recording")
            st.audio(r["input_audio"])

        for title, value in [
            ("Detected Language", r["language"]),
            ("Original Transcription", r["transcript"]),
            ("English Translation", r["translated_question"]),
            ("Domain", r["domain"].capitalize()),
        ]:
            st.markdown(f"## {title}")
            st.markdown(bob_box(value, "blue"), unsafe_allow_html=True)

        st.markdown("## Keywords")
        st.dataframe(
            pd.DataFrame({
                f"Native keyword ({r['language'].title()})": [r["keyword_native"][k] for k in r["keywords"]],
                "English keyword": r["keywords"],
            }),
            hide_index=True,
            width="stretch",
        )

    with tab3:
        st.markdown("## Matched Products")
        if r["match_mode"] == "matched":
            st.caption(
                f"Showing the top {len(r['matched'])} of {r['catalogue_size']} "
                f"{r['domain'].title()} products, ranked by keyword hits."
            )
        else:
            st.warning(
                "No product matched your keywords, so the first "
                f"{len(r['matched'])} products were used instead."
            )

        st.dataframe(
            r["matched"],
            hide_index=True,
            width="stretch",
            column_config={
                "Keyword hits": st.column_config.ProgressColumn(
                    "Keyword hits",
                    min_value=0,
                    max_value=max(1, len(r["keywords"])),
                    format="%d",
                )
            },
        )
        st.download_button(
            "⬇️ Download matched products (CSV)",
            r["matched"].to_csv(index=False).encode("utf-8"),
            file_name=f"matched_{'_'.join(r['domain'].split())}.csv",
            mime="text/csv",
            key=f"dl_csv_{r['id']}",
        )

    with tab4:
        st.markdown("## Model Response")
        st.markdown(bob_box(r["answer_en"]), unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download answer (.txt)",
            r["answer_en"],
            file_name="bob_answer_english.txt",
            key=f"dl_en_{r['id']}",
        )

    with tab5:
        st.markdown("## Translated Response")
        st.markdown(bob_box(r["answer_local"]), unsafe_allow_html=True)

        st.markdown("## Spoken Response")
        if r["tts"]:
            # Native audio element, keyed per answer, so a new answer always
            # loads its own voice instead of reusing the previous one.
            with st.container(key=f"voice_{r['id']}"):
                st.audio(
                    r["tts"],
                    format="audio/mp3",
                    autoplay=bool(AUTOPLAY and not r["played"]),
                )
            r["played"] = True

            st.download_button(
                "⬇️ Download spoken answer (.mp3)",
                r["tts"],
                file_name="bob_answer.mp3",
                mime="audio/mpeg",
                key=f"dl_mp3_{r['id']}",
            )
        else:
            st.warning(
                "Voice could not be generated, but the written answer above is complete. "
                f"({r.get('tts_error') or 'unknown error'})"
            )

    st.markdown("**Was this answer helpful?**")
    rating = st.feedback("thumbs", key=f"fb_{r['id']}")
    if rating is not None:
        for h in st.session_state.history:
            if h["id"] == r["id"]:
                if h["feedback"] != rating:  # toast only when the choice changes
                    bg, fg, border = (
                        ("#e6f6ec", "#1e7e34", "#28a745") if rating == 1
                        else ("#fdecee", "#b02a37", "#dc3545")
                    )
                    st.markdown(
                        f"""<style>
                        div[data-testid="stToast"] {{
                            background: {bg} !important;
                            border-left: 5px solid {border} !important;
                        }}
                        div[data-testid="stToast"] [data-testid="stMarkdownContainer"] p,
                        div[data-testid="stToast"] [data-testid="stMarkdownContainer"] span,
                        div[data-testid="stToast"] [data-testid="stMarkdownContainer"] strong {{
                            color: {fg} !important;
                            font-weight: 600;
                        }}
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    if rating == 1:
                        st.toast("Thanks! Glad this was helpful.", icon="👍")
                    else:
                        st.toast("Sorry about that. We'll try to do better.", icon="👎")
                h["feedback"] = rating


def render_history():
    history = st.session_state.history
    if not history:
        st.info("Your questions and answers will appear here during this session.")
        return

    icons = {1: "👍", 0: "👎", None: "—"}
    table = pd.DataFrame([
        {
            "Time": h["time"],
            "Input": h["mode"],
            "Language": h["language"].title(),
            "Category": h["domain"].title(),
            "Question (English)": h["question"],
            "Seconds": round(h["total"], 1),
            "Feedback": icons[h["feedback"]],
        }
        for h in reversed(history)
    ])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Download history (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="bob_session_history.csv",
        mime="text/csv",
    )

    st.markdown("## Review an earlier answer")
    chosen = st.selectbox(
        "Choose a question",
        list(reversed(history)),
        format_func=lambda h: f"{h['time']} — {h['question'][:70]}",
        label_visibility="collapsed",
    )
    st.markdown("**English answer**")
    st.markdown(bob_box(chosen["answer_en"]), unsafe_allow_html=True)
    st.markdown(f"**Answer in {chosen['language'].title()}**")
    st.markdown(bob_box(chosen["answer_local"]), unsafe_allow_html=True)


def render_analytics():
    history = st.session_state.history
    if not history:
        st.info("Ask a question to see session insights.")
        return

    left, right = st.columns(2)

    with left:
        st.markdown("### Questions by product category")
        by_domain = pd.Series(Counter(h["domain"].title() for h in history))
        st.bar_chart(by_domain, color="#ff7a00")

    with right:
        st.markdown("### Questions by language")
        by_language = pd.Series(Counter(h["language"].title() for h in history))
        st.bar_chart(by_language, color="#0059b3")

    left, right = st.columns(2)

    with left:
        st.markdown("### Average time per stage (seconds)")
        stage_totals = Counter()
        for h in history:
            stage_totals.update(h["timings"])
        stage_avg = pd.Series({k: v / len(history) for k, v in stage_totals.items()})
        st.bar_chart(stage_avg, color="#ff7a00")

    with right:
        st.markdown("### What customers are asking about")
        pair_counts = Counter(
            (h["keyword_native"].get(k, k), k, h["language"].title())
            for h in history
            for k in h["keywords"]
        )
        if pair_counts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Native keyword": native, "English keyword": english,
                         "Language": language, "Asked": n}
                        for (native, english, language), n in pair_counts.most_common(20)
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No keywords captured yet.")


def render_about():
    st.markdown("## Tips for better answers")
    st.markdown(
        "- Speak close to the microphone and pause when you finish.\n"
        "- Mention the product and what matters to you (amount, tenure, salary, purpose).\n"
        "- Check the playback, then press Get answer. Press New Recording to record again.\n"
        "- Browsers only allow microphone access on `localhost` or HTTPS pages."
    )
    st.markdown("## Privacy")
    st.markdown(
        "Recordings are sent to Google Gemini to be understood, and the answer text is sent to "
        "Google Text-to-Speech to be spoken. Nothing is written to disk. Your session history is kept in "
        "the app's memory under a random link id so it survives a page refresh; use "
        "'Clear history' in the sidebar to remove it, and it is lost whenever the app restarts."
    )
    st.markdown("## Please note")
    st.markdown(
        "Answers are AI-generated from the product catalogue and are for guidance only. "
        "Rates, eligibility and charges can change — confirm the final terms with your BOB branch."
    )


# =====================================================
# INPUT  (speak or type, all in one place)
# =====================================================


def speak_panel():
    if st.session_state.audio_data:
        st.audio(st.session_state.audio_data)

        col_ask, col_new = st.columns(2)
        if col_ask.button("✅ Get answer", type="primary", key="ask_audio"):
            st.session_state.confirmed_audio = True
            rerun()
        if col_new.button("🆕 New Recording", key="new_recording"):
            st.session_state.audio_data = None
            st.session_state.confirmed_audio = False
            rerun()

        st.caption(
            "This recording stays here until you press New Recording, "
            "so you can ask again or try a different model."
        )
        return

    st.caption("Click the microphone and speak. Pause for 2 seconds and it stops by itself.")
    recorded = audio_recorder(
        text="🎤 Click to Start / Stop Recording",
        recording_color="#e74c3c",
        neutral_color="#ff7a00",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=SAMPLE_RATE,
    )

    if recorded:
        if len(recorded) < MIN_AUDIO_BYTES:
            st.warning("⚠ No voice detected. Please record again.")
        else:
            st.session_state.audio_data = recorded
            st.session_state.confirmed_audio = AUTO_SUBMIT
            rerun()


def type_panel():
    with st.form("typed_question", clear_on_submit=True):
        typed = st.text_area(
            "Your question",
            placeholder="e.g. I want an education loan for studying abroad",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask BOB AI", type="primary")

    if submitted and typed.strip():
        st.session_state.pending_text = typed.strip()
        rerun()

    st.caption("Or tap an example:")
    for col, sample in zip(st.columns(len(SAMPLE_QUESTIONS)), SAMPLE_QUESTIONS):
        if col.button(sample, key=f"sample_{sample}"):
            st.session_state.pending_text = sample
            rerun()


def input_card():
    with st.container(border=True):
        st.markdown("### 🎙️ Ask your question")
        mode = st.segmented_control(
            "How would you like to ask?",
            ["🎤 Speak", "⌨️ Type"],
            default="🎤 Speak",
            key="input_mode",
            label_visibility="collapsed",
        )
        st.caption("Works in English, Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Malayalam and Gujarati.")

        if mode == "⌨️ Type":
            type_panel()
        else:
            speak_panel()


# =====================================================
# DEVELOPER OPTIONS - data downloader
# Scrapes the latest product pages from the bank website so the
# Bank_Product/*.xlsx files can be refreshed. (Logic from scrape.ipynb)
# =====================================================

REQUEST_DELAY = 0.3  # seconds between page requests, to stay polite to the site
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bankofbaroda.in/",
    "Connection": "keep-alive",
}

# category -> (listing page, how sure we are). The developer can edit these in the UI.
SOURCE_URLS = {
    "home loan": ("https://bankofbaroda.bank.in/loans/home-loan", "site"),
    "vehicle loan": ("https://bankofbaroda.bank.in/loans/vehicle-loan", "verify"),
    "education loan": ("https://bankofbaroda.bank.in/loans/education-loan", "notebook"),
    "personal loan": ("https://bankofbaroda.bank.in/loans/personal-loan", "verify"),
    "mortgage loan": ("https://bankofbaroda.bank.in/loans/mortgage-loan", "verify"),
    "term deposits": ("https://bankofbaroda.bank.in/personal-banking/accounts/term-deposit", "verify"),
    "savings account": ("https://bankofbaroda.bank.in/personal-banking/accounts/savings-accounts", "verify"),
    "salary account": ("https://bankofbaroda.bank.in/personal-banking/accounts/salary-accounts", "verify"),
    "current account": ("https://bankofbaroda.bank.in/personal-banking/accounts/current-accounts", "verify"),
    "debit card": ("https://bankofbaroda.bank.in/digital-products/cards/debit-cards", "notebook"),
}

SOURCE_NOTE = {
    "notebook": "✅ from your notebook",
    "site": "✅ page found on the bank site",
    "verify": "⚠️ best guess, please verify",
}

# columns used by the card-style pages (the notebook's fallback branch)
CARD_SECTIONS = [
    "benefits", "features", "eligible_customer_segments",
    "fees_charges", "card_limits", "dos_and_donts",
]


def normalize_heading(heading: str) -> str:
    if not heading:
        return None
    heading = heading.strip()
    heading = heading.lower()

    # Replace common variations
    heading = heading.replace("&", "and")
    heading = heading.replace("’", "'")

    # Collapse multiple spaces
    heading = re.sub(r"\s+", " ", heading)

    # Map known variants
    mapping = {
        "fees/charges": "Fees and Charges",
        "fees & charges": "Fees and Charges",
        "fees_charges": "Fees and Charges",
        "fees and charges": "Fees and Charges",
        "do's and don'ts": "Do's and Don'ts",
        "do's & don'ts": "Do's and Don'ts",
        "dos_and_donts": "Do's and Don'ts",
        "most important terms and conditions (mitc)": "Terms and Conditions",
        "terms & conditions": "Terms and Conditions",
        "terms and conditions": "Terms and Conditions",
        "eligibility criteria": "Eligibility",
        "steps to apply": "Steps to Apply",
        "documents required": "Documents Required",
        "audio guide": "Audio Guide",
        "benefits": "Benefits",
        "features": "Features",
        "eligible_customer_segments": "Eligibility",
        "card_limits": "Card Limits",
    }

    return mapping.get(heading, heading.title())


def _table_to_text(table):
    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" : ".join(cells))
    return "\n".join(rows)


def parse_product_page(soup, link, base_url):
    """One product page -> one-row DataFrame (same logic as the notebook)."""
    tabs = soup.find_all("div", class_="tab")

    tab_contents = {"Card": link.replace(base_url, ""), "URL": link}

    for tab in tabs:
        h3 = tab.find("h3")
        heading = h3.get_text(strip=True) if h3 else None
        if heading and ":" in heading:
            tab_contents["Card"] = heading.split(":", 1)[0].strip()
            heading = heading.split(":", 1)[1].strip()

        heading = normalize_heading(heading)
        if not heading:  # a tab without a heading has nothing to label it with
            continue

        # Case 1: Table present
        table = tab.find("table")
        if table:
            tab_contents[heading] = _table_to_text(table)

        # Case 2: List items or paragraphs
        else:
            texts = [li.get_text(" ", strip=True) for li in tab.find_all("li")]
            texts += [p.get_text(" ", strip=True) for p in tab.find_all("p")]
            tab_contents[heading] = "\n".join(texts)

    if len(tab_contents) <= 2:
        title = soup.find("title")
        tab_contents["Card"] = title.text.strip() if title else tab_contents["Card"]

        # Collect all elements with an id (both section and div)
        for block in soup.find_all(["section", "div"], id=True):
            block_id = block.get("id")
            block_data = []

            for table in block.find_all("table"):
                text = _table_to_text(table)
                if text:
                    block_data.append(text)

            for li in block.find_all("li"):
                block_data.append(li.get_text(" ", strip=True))
            for p in block.find_all("p"):
                block_data.append(p.get_text(" ", strip=True))

            if block_data:
                tab_contents[block_id] = "\n".join(block_data)

        df = pd.DataFrame([tab_contents])
        if any(c in tab_contents for c in CARD_SECTIONS):
            df = df.reindex(columns=["Card", "URL"] + CARD_SECTIONS)
            df.columns = list(df.columns[:2]) + [normalize_heading(c) for c in df.columns[2:]]
    else:
        df = pd.DataFrame([tab_contents])

    # tidy repeated spaces (skips empty cells, which are not text)
    return df.apply(
        lambda col: col.map(lambda v: re.sub(r" +", " ", v) if isinstance(v, str) else v)
    )


def scrape_category(url, on_event=None, delay=REQUEST_DELAY):
    """Scrape every product page linked from a category page.

    Returns (DataFrame, failed) where failed is a list of (url, error).
    on_event(kind, **info) is called so the UI can show progress.
    """
    import requests
    from bs4 import BeautifulSoup

    emit = on_event or (lambda *args, **kwargs: None)

    emit("fetch_listing", url=url)
    response = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all product items
    all_links = []
    for item in soup.find_all("li", class_="item"):
        for a in item.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if url not in href:
                continue
            if a["href"] != "#":  # skip dummy links
                all_links.append(href)

    links = list(dict.fromkeys(all_links))  # unique, keeps page order
    emit("links", total=len(links))

    frames, failed = [], []
    for n, link in enumerate(links, 1):
        emit("page_start", index=n, total=len(links), url=link)
        try:
            page = requests.get(link, headers=SCRAPE_HEADERS, timeout=30)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, "html.parser")
            frames.append(parse_product_page(page_soup, link, url))
            emit("page_done", index=n, total=len(links), url=link)
        except Exception as e:  # skip the page, but tell the developer
            failed.append((link, str(e)))
            emit("page_failed", index=n, total=len(links), url=link, error=str(e))
        time.sleep(delay)

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return final_df, failed


def df_to_xlsx_bytes(df):
    clean = df.apply(
        lambda col: col.map(lambda v: ILLEGAL_XLSX_CHARS.sub("", v) if isinstance(v, str) else v)
    )
    buf = io.BytesIO()
    clean.to_excel(buf, index=False)
    return buf.getvalue()


def build_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files.values():
            zf.writestr(f["name"], f["bytes"])
    return buf.getvalue()


def short_url(url, width=70):
    return url if len(url) <= width else "…" + url[-(width - 1):]


def run_download(selected, urls):
    """Scrape the chosen categories, showing live progress."""
    total = len(selected)
    started = time.perf_counter()
    done_ok = 0

    overall = st.progress(0.0, text=f"0 of {total} categories done")
    status = st.status("Downloading the latest data…", expanded=True)

    with status:
        page_bar = st.progress(0.0, text="Starting…")

        for k, domain in enumerate(selected):
            url = urls[domain].strip()
            label = domain.title()
            st.write(f"🔎 **{label}**  ·  {short_url(url)}")
            page_bar.progress(0.0, text=f"{label}: opening the listing page…")

            def on_event(kind, label=label, **info):
                if kind == "links":
                    page_bar.progress(0.0, text=f"{label}: found {info['total']} product pages")
                elif kind == "page_start":
                    page_bar.progress(
                        (info["index"] - 1) / info["total"],
                        text=f"{label}: page {info['index']} of {info['total']}  ·  {short_url(info['url'], 55)}",
                    )
                elif kind == "page_done":
                    page_bar.progress(
                        info["index"] / info["total"],
                        text=f"{label}: page {info['index']} of {info['total']} done",
                    )
                elif kind == "page_failed":
                    st.write(f"⚠️ Skipped {short_url(info['url'], 60)} ({info['error']})")

            t0 = time.perf_counter()
            try:
                df, failed = scrape_category(url, on_event)
            except Exception as e:
                st.error(f"{label}: could not open {url}. {e}")
                overall.progress((k + 1) / total, text=f"{k + 1} of {total} categories done")
                continue

            if df.empty:
                st.warning(
                    f"{label}: no product pages were read. Check the source URL "
                    "(the listing page must contain links inside <li class='item'>)."
                )
                overall.progress((k + 1) / total, text=f"{k + 1} of {total} categories done")
                continue

            data = df_to_xlsx_bytes(df)
            st.session_state.dev_files[domain] = {
                "name": f"{'_'.join(domain.split())}.xlsx",
                "bytes": data,
                "df": df,
                "rows": len(df),
                "cols": df.shape[1],
                "url": url,
                "failed": failed,
                "seconds": time.perf_counter() - t0,
                "scraped_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            done_ok += 1
            st.write(f"✅ {label}: {len(df)} products saved" + (f", {len(failed)} skipped" if failed else ""))
            overall.progress((k + 1) / total, text=f"{k + 1} of {total} categories done")

        elapsed = time.perf_counter() - started
        status.update(
            label=f"Finished: {done_ok} of {total} categories in {elapsed:.0f}s",
            state="complete" if done_ok else "error",
            expanded=False,
        )

    if done_ok:
        st.toast("Download complete", icon="✅")


def render_dev_files():
    files = st.session_state.dev_files
    if not files:
        st.info("No files yet. Choose categories above and press **Start download**.")
        return

    st.markdown("## 📦 Ready to download")

    for domain, f in files.items():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3.2, 2.4, 2.2, 1.8])
            c1.markdown(f"📄 **{f['name']}**")
            c1.caption(f"{domain.title()} · {f['scraped_at']}")
            c2.markdown(
                chips([f"{f['rows']} products", f"{f['cols']} columns"]),
                unsafe_allow_html=True,
            )
            size_kb = len(f["bytes"]) / 1024
            note = f"{size_kb:.0f} KB · {f['seconds']:.0f}s"
            if f["failed"]:
                note += f" · ⚠️ {len(f['failed'])} skipped"
            c3.caption(note)
            c4.download_button(
                "⬇️ Download",
                f["bytes"],
                file_name=f["name"],
                mime=XLSX_MIME,
                key=f"dev_dl_{domain}",
            )

    left, right = st.columns([2, 1])
    left.download_button(
        f"🗜️ Download all {len(files)} files (ZIP)",
        build_zip(files),
        file_name="bob_product_data.zip",
        mime="application/zip",
        key="dev_dl_zip",
        type="primary",
    )
    if right.button("🗑 Clear downloaded files", key="dev_clear_files"):
        st.session_state.dev_files = {}
        rerun()

    skipped = {d: f["failed"] for d, f in files.items() if f["failed"]}
    if skipped:
        with st.expander("⚠️ Pages that were skipped"):
            for d, items in skipped.items():
                st.markdown(f"**{d.title()}**")
                for link, err in items:
                    st.caption(f"{link}  —  {err}")

    st.markdown("### 👁️ Preview")
    chosen = st.selectbox(
        "Preview a file",
        list(files),
        format_func=str.title,
        label_visibility="collapsed",
        key="dev_preview",
    )
    st.dataframe(files[chosen]["df"], hide_index=True, width="stretch")


def render_dev_page():
    head, back = st.columns([5, 1])
    head.markdown("## 📥 Data Downloader")
    if back.button("⬅️ Back", key="dev_back_top"):
        st.session_state.dev_open = False
        rerun()

    st.caption(
        "Fetch the latest product data from the bank website, review it, then replace "
        "the matching files in the Bank_Product/ folder of your repository."
    )

    try:
        import bs4  # noqa: F401
        import requests  # noqa: F401
    except ImportError:
        st.error("Missing packages. Run `pip install requests beautifulsoup4` (they are in requirements.txt).")
        return

    selected = st.multiselect(
        "Categories to download",
        DOMAINS,
        key="dev_selected",
        format_func=str.title,
        placeholder="Choose one or more categories…",
    )

    col_all, col_none, _ = st.columns([1, 1, 4])
    col_all.button(
        "Select all", key="dev_select_all",
        on_click=lambda: st.session_state.update(dev_selected=list(DOMAINS)),
    )
    col_none.button(
        "Clear", key="dev_select_none",
        on_click=lambda: st.session_state.update(dev_selected=[]),
    )

    urls = {}
    with st.expander("🔗 Source pages (editable)", expanded=False):
        if not selected:
            st.caption("Pick a category to see and edit its source page.")
        for domain in selected:
            default_url, level = SOURCE_URLS[domain]
            urls[domain] = st.text_input(
                f"{domain.title()}  ·  {SOURCE_NOTE[level]}",
                value=default_url,
                key=f"dev_url_{domain}",
            )

    if st.button(
        f"🚀 Start download ({len(selected)} selected)" if selected else "🚀 Start download",
        type="primary",
        disabled=not selected,
        key="dev_start",
    ):
        st.caption("Keep this tab open until it finishes.")
        run_download(selected, urls)

    render_dev_files()

    st.info(
        "After downloading, edit the Excel files if you want, then put them in the "
        "**Bank_Product/** folder of your repo (same file names) and commit. "
        "The assistant reads them from there."
    )


# =====================================================
# MAIN LAYOUT
# =====================================================

# Developer page replaces the assistant while it is open
if st.session_state.dev_unlocked and st.session_state.dev_open:
    render_dev_page()
    persist()
    st.stop()

kpi_slot = st.empty()

input_card()

# ---- decide what to process on this run ----
job = None
if st.session_state.confirmed_audio and st.session_state.audio_data:
    job = {"kind": "audio", "audio": st.session_state.audio_data}
    st.session_state.confirmed_audio = False  # the recording itself is kept
elif st.session_state.pending_text:
    job = {"kind": "text", "text": st.session_state.pending_text}
    st.session_state.pending_text = None

tab_assistant, tab_history, tab_analytics, tab_about = st.tabs([
    "💬 Assistant", "🕘 History", "📊 Insights", "ℹ️ About",
])

with tab_assistant:
    if job:
        with st.status("Working on your request…", expanded=True) as status:
            try:
                result = run_pipeline(job, status)
            except PipelineError as e:
                status.update(label="Could not complete the request", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(f"Something went wrong: {e}")
                print(e)
            else:
                status.update(
                    label=f"Answer ready in {result['total']:.1f}s",
                    state="complete",
                    expanded=False,
                )
                st.session_state.current = result
                st.session_state.history.append({
                    "id": result["id"],
                    "time": result["time"],
                    "mode": "🎤 Voice" if result["kind"] == "audio" else "⌨️ Typed",
                    "language": result["language"],
                    "domain": result["domain"],
                    "question": result["translated_question"],
                    "keywords": result["keywords"],
                    "keyword_native": result["keyword_native"],
                    "answer_en": result["answer_en"],
                    "answer_local": result["answer_local"],
                    "timings": result["timings"],
                    "total": result["total"],
                    "feedback": None,
                })
                st.toast("Answer ready", icon="✅")

    if st.session_state.current:
        render_result(st.session_state.current)
    elif not job:
        render_welcome()

with tab_history:
    render_history()

with tab_analytics:
    render_analytics()

with tab_about:
    render_about()

# KPIs are filled last so they include the answer produced on this run
render_kpis(kpi_slot)
persist()
