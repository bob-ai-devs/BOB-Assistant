# speech.py
# BOB AI Multilingual Product Assistant
# Record / type → Understand → Match products → Answer → Speak

import ast
import base64
import difflib
import html
import io
import os
import re
import textwrap
import time
from collections import Counter
from datetime import datetime

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

/* ===== GLOBAL ===== */
.stApp {
    background: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== HEADINGS ===== */
h1, h2, h3, h4 {
    color: #002e6e !important;
    font-weight: 700;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffa64d, #ff8c1a);
}

/* Sidebar text only */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* ===== BUTTONS ===== */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #0059b3;
    color: white;
    border-radius: 6px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #003f80;
    color: white;
}

/* ===== INFO / SUCCESS / WARNING / ERROR ===== */
div[data-testid="stAlert"] {
    border-radius: 10px;
    padding: 12px;
}

/* INFO */
div[data-testid="stAlert"][kind="info"] {
    background-color: #ffe6cc !important;
    border-left: 5px solid #ff8c1a !important;
}

div[data-testid="stAlert"][kind="info"] * {
    color: black !important;
}

/* SUCCESS */
div[data-testid="stAlert"][kind="success"] {
    background-color: #d4edda !important;
    border-left: 5px solid #28a745 !important;
}

div[data-testid="stAlert"][kind="success"] * {
    color: black !important;
}

/* WARNING */
div[data-testid="stAlert"][kind="warning"] {
    background-color: #fff3cd !important;
    border-left: 5px solid #ffc107 !important;
}

div[data-testid="stAlert"][kind="warning"] * {
    color: black !important;
}

/* ERROR */
div[data-testid="stAlert"][kind="error"] {
    background-color: #f8d7da !important;
    border-left: 5px solid #dc3545 !important;
}

div[data-testid="stAlert"][kind="error"] * {
    color: black !important;
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border-radius: 6px;
}

/* Dropdown items */
div[role="listbox"] {
    color: black !important;
}

/* ===== TABS ===== */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #002e6e;
    background-color: #f5f7fb;
    border-radius: 6px 6px 0 0;
    margin-right: 3px;
    padding: 4px 10px;
    font-size: 13px;
    transition: all 0.2s ease-in-out;
}

button[data-baseweb="tab"]:hover {
    background-color: #e6f0ff;
    color: #0059b3;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ff7a00 !important;
    color: white !important;
    border-bottom: 2px solid #ff7a00;
}

div[data-baseweb="tab-list"] {
    border-bottom: 1px solid #eee;
}

div[data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* ===== MARKDOWN TEXT ===== */
p, li, span {
    color: black;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    height: 6px;
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #0059b3;
    border-radius: 10px;
}

/* ===== MARKDOWN TEXT ===== */
/* Removes odd indentation/margin on first line */
[data-testid="stMarkdownContainer"] p {
    margin-top: 0rem !important;
    margin-bottom: 0.6rem !important;
    padding-left: 0rem !important;
    color: #1f1f1f !important;
    line-height: 1.6;
    font-size: 15px;
}

/* Markdown lists */
[data-testid="stMarkdownContainer"] li {
    color: #1f1f1f !important;
    margin-bottom: 0.25rem;
}

/* Bold text */
[data-testid="stMarkdownContainer"] strong {
    color: #ff7a00 !important;
    font-weight: 700;
}

/* Inline code */
[data-testid="stMarkdownContainer"] code {
    background-color: #fff4e6;
    color: #d35400;
    padding: 2px 5px;
    border-radius: 5px;
}

/* Links */
[data-testid="stMarkdownContainer"] a {
    color: #0059b3 !important;
    text-decoration: none;
    font-weight: 500;
}

/* ===== BOB SHADOW BOX ===== */
.bob-box {
    background: #fff7ed;              /* very light BOB orange */
    border-left: 5px solid #ff7a00;  /* BOB orange accent */
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 10px;
    margin-bottom: 15px;
    color: #1f1f1f;
    line-height: 1.6;
    font-size: 15px;
}

.bob-box p {
    margin: 0;
    line-height: 1.6;
    font-size: 15px;
}

.bob-box strong {
    color: #ff7a00;
}

/* ===== DASHBOARD ADDITIONS ===== */

/* KPI cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e3eaf6;
    border-top: 4px solid #ff7a00;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    height: 100%;
}

.kpi-card.blue { border-top-color: #0059b3; }

.kpi-label { color: #5b6b82; font-size: 13px; font-weight: 600; }
.kpi-value { color: #002e6e; font-size: 28px; font-weight: 700; line-height: 1.25; }
.kpi-sub   { color: #7a8699; font-size: 12px; min-height: 16px; }

/* Chips */
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
    background: #fff4e6;
    color: #d35400;
    border-color: #ffcf99;
}

/* Pipeline stage strip */
.stage-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px 0; }

.stage {
    background: #f5f7fb;
    border: 1px solid #dbe5f3;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 600;
    color: #002e6e;
}

.stage b { color: #ff7a00; }

/* How-it-works cards */
.how-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 8px 0 18px 0;
}

.how-card {
    background: #f5f7fb;
    border: 1px solid #dbe5f3;
    border-radius: 12px;
    padding: 14px;
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

.how-title { color: #002e6e; font-weight: 700; font-size: 15px; }
.how-text  { color: #4a5a70; font-size: 13px; line-height: 1.45; }

/* Fallback banner when the image file is missing */
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


def bob_box(text) -> str:
    return (
        '<div class="bob-box" style="padding: 1rem; border-radius: 10px; '
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

    # return sorted(voice_models), sorted(gen_models)
    return ["gemini-flash-lite-latest"], []


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
    "allow_recording": False,
    "pending_text": None,
    "current": None,
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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

TOP_N = st.sidebar.slider(
    "🏦 Products to consider", min_value=3, max_value=30, value=10,
    help="How many matching products are sent to the model to build the answer.",
)

st.sidebar.markdown("## 🎤 Recorder")

if st.sidebar.button("🆕 New Recording"):
    st.session_state.audio_data = None
    st.session_state.confirmed_audio = False
    st.session_state.allow_recording = True
    st.rerun()

if not st.session_state.allow_recording and not st.session_state.audio_data:
    st.sidebar.info("Click 'New Recording' to start.")

elif st.session_state.allow_recording:
    with st.sidebar:
        recorded = audio_recorder(
            text="🎤 Click to Start / Stop Recording",
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=SAMPLE_RATE,
        )

    if recorded:
        if len(recorded) < MIN_AUDIO_BYTES:
            st.sidebar.warning("⚠ No voice detected.\n\nPlease record again.")
            st.session_state.audio_data = None
        else:
            st.sidebar.success("✅ Voice detected successfully")
            st.session_state.audio_data = recorded
            st.session_state.allow_recording = False
            st.rerun()

if st.session_state.audio_data:
    st.sidebar.markdown("## 🎧 Recorded Audio")
    st.sidebar.audio(st.session_state.audio_data)
    st.sidebar.info("Please verify the recorded voice input.")

    if st.sidebar.button("✅ Continue with this Audio"):
        st.session_state.confirmed_audio = True
        st.rerun()

st.sidebar.markdown("## 🧹 Session")

if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.session_state.current = None
    st.rerun()

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


def stream_text(model, contents, placeholder):
    text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents):
        if chunk.text:
            text += chunk.text
            placeholder.markdown(bob_box(text), unsafe_allow_html=True)
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

    raw_extract = stream_text(MODEL, contents, placeholder)
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
            kpi_card("Questions answered", n, "this session"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            kpi_card("Languages used", len(languages), ", ".join(languages[:3]) or "—", "blue"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            kpi_card("Avg. response time", f"{avg:.1f}s" if avg else "—", "question to spoken answer"),
            unsafe_allow_html=True,
        )
        c4.markdown(
            kpi_card("Helpful answers", helpful, f"{len(rated)} rated" if rated else "rate an answer below", "blue"),
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

    st.info("Use **🆕 New Recording** in the sidebar, or type a question above, to get started.")


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
        st.markdown(bob_box(r["raw_extract"]), unsafe_allow_html=True)

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
            st.markdown(bob_box(value), unsafe_allow_html=True)

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
            if AUTOPLAY and not r["played"]:
                audio_b64 = base64.b64encode(r["tts"]).decode()
                st.markdown(
                    '<audio autoplay controls style="width:100%">'
                    f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">'
                    "</audio>",
                    unsafe_allow_html=True,
                )
            else:
                st.audio(r["tts"], format="audio/mp3")
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
        "- Check the playback before continuing, and re-record if it sounds unclear.\n"
        "- Browsers only allow microphone access on `localhost` or HTTPS pages."
    )
    st.markdown("## Privacy")
    st.markdown(
        "Recordings are sent to Google Gemini to be understood, and the answer text is sent to "
        "Google Text-to-Speech to be spoken. Nothing is stored on the server; history lives only "
        "in your browser session and disappears when you close the tab."
    )
    st.markdown("## Please note")
    st.markdown(
        "Answers are AI-generated from the product catalogue and are for guidance only. "
        "Rates, eligibility and charges can change — confirm the final terms with your BOB branch."
    )


# =====================================================
# MAIN LAYOUT
# =====================================================

kpi_slot = st.empty()

with st.expander("⌨️ Prefer typing? Ask your question in any supported language"):
    with st.form("typed_question", clear_on_submit=True):
        typed = st.text_area(
            "Your question",
            placeholder="e.g. I want an education loan for studying abroad",
            height=80,
            label_visibility="collapsed",
        )
        if st.form_submit_button("Ask BOB AI") and typed.strip():
            st.session_state.pending_text = typed.strip()
            st.rerun()

    st.caption("Or try an example:")
    for col, sample in zip(st.columns(len(SAMPLE_QUESTIONS)), SAMPLE_QUESTIONS):
        if col.button(sample, key=f"sample_{sample}"):
            st.session_state.pending_text = sample
            st.rerun()

# ---- decide what to process on this run ----
job = None
if st.session_state.confirmed_audio and st.session_state.audio_data:
    job = {"kind": "audio", "audio": st.session_state.audio_data}
    st.session_state.confirmed_audio = False
    st.session_state.audio_data = None
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
