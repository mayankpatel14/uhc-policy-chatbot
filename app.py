"""Streamlit web interface for the UHC medical policy chatbot."""

import time
import streamlit as st

from chatbot.config import GROQ_MODEL, RETRIEVAL_TOP_K, MAX_HISTORY_TURNS
from chatbot.retriever import PolicyRetriever
from chatbot.llm_groq import GroqClient, GroqError
from chatbot.prompts import format_context, build_messages, deduplicate_chunks

st.set_page_config(
    page_title="UHC Policy Chatbot",
    page_icon="🏥",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_retriever() -> PolicyRetriever:
    logs: list[str] = []
    r = PolicyRetriever()
    r.init(status_callback=lambda msg: logs.append(msg))
    return r


@st.cache_resource
def load_llm() -> GroqClient:
    return GroqClient()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 UHC Policy Chatbot")
    st.caption(f"LLM: **{GROQ_MODEL}** via Groq")
    st.caption(f"Retrieval: **MedEmbed** → Qdrant (top-{RETRIEVAL_TOP_K})")
    st.divider()

    st.markdown("### How to use")
    st.markdown(
        "Ask questions about UnitedHealthcare medical policies — "
        "coverage criteria, CPT codes, medical necessity, and more."
    )
    st.markdown(
        "**Examples:**\n"
        "- Is bariatric surgery covered for BMI over 40?\n"
        "- What documentation is needed for gender-affirming surgery?\n"
        "- Is HFCWO covered for cystic fibrosis?\n"
        "- What are the criteria for whole genome sequencing?"
    )
    st.divider()

    tts_enabled = st.toggle("🔊 Read answers aloud", value=False)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.chunks_history = []
        st.rerun()

    st.caption("Answers are based on official UHC policy documents only.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks_history" not in st.session_state:
    st.session_state.chunks_history = []

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
with st.spinner("Loading MedEmbed model and connecting to Qdrant..."):
    retriever = load_retriever()

try:
    llm = load_llm()
except GroqError as e:
    st.error(f"LLM initialization failed: {e}")
    st.stop()

st.title("🏥 UHC Medical Policy Chatbot")
st.caption(
    "Ask questions about UnitedHealthcare insurance policies. "
    "Answers are grounded in official policy documents with source citations."
)

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and i // 2 < len(st.session_state.chunks_history):
            chunks_for_msg = st.session_state.chunks_history[i // 2]
            if chunks_for_msg:
                with st.expander("📚 Sources", expanded=False):
                    for c in chunks_for_msg:
                        st.markdown(
                            f"- **[{c.score:.2f}]** `{c.policy_name}` — "
                            f"{c.section} *(pages {c.page_start}–{c.page_end})*"
                        )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if query := st.chat_input("Ask about UHC medical policies..."):
    query = query.strip()

    if not query:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            t0 = time.perf_counter()
            try:
                chunks = retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
            except RuntimeError as e:
                st.error(f"Retrieval error: {e}")
                st.stop()
            t_retrieval = time.perf_counter() - t0

        if not chunks:
            response_text = (
                "I don't have enough policy information to answer this question. "
                "Try rephrasing or asking about a specific UHC policy topic."
            )
            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            st.session_state.chunks_history.append([])
            st.stop()

        context = format_context(chunks)

        history_for_llm = []
        turns = st.session_state.messages[:-1]
        if len(turns) > MAX_HISTORY_TURNS * 2:
            turns = turns[-(MAX_HISTORY_TURNS * 2):]
        for m in turns:
            history_for_llm.append({"role": m["role"], "content": m["content"]})

        messages = build_messages(query, context, history=history_for_llm)

        try:
            t1 = time.perf_counter()
            response_text = st.write_stream(llm.chat_stream(messages))
            t_gen = time.perf_counter() - t1
        except GroqError as e:
            st.error(str(e))
            st.stop()

        deduped = deduplicate_chunks(chunks)
        with st.expander("📚 Sources", expanded=False):
            for c in deduped:
                st.markdown(
                    f"- **[{c.score:.2f}]** `{c.policy_name}` — "
                    f"{c.section} *(pages {c.page_start}–{c.page_end})*"
                )

        st.caption(
            f"Retrieval: {t_retrieval:.1f}s · Generation: {t_gen:.1f}s"
        )

        if tts_enabled and response_text:
            with st.spinner("Generating audio..."):
                try:
                    from chatbot.tts import synthesize
                    audio_bytes = synthesize(response_text)
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                except Exception as e:
                    st.caption(f"⚠️ TTS unavailable: {e}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
    st.session_state.chunks_history.append(deduped)
