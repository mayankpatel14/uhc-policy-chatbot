# UHC Medical Policy Chatbot

A RAG-powered chatbot that answers questions about UnitedHealthcare (UHC) medical policies. Built for doctors, hospital staff, and insurance coordinators who need accurate, cited answers about coverage criteria, CPT/HCPCS codes, and medical necessity requirements.

**Live demo:** [https://huggingface.co/spaces/mxp1404/uhc-policy-chatbot](https://huggingface.co/spaces/mxp1404/uhc-policy-chatbot)

### How to Use

1. Open the link above in your browser.
2. Wait for the model to load (first visit takes ~30s for MedEmbed to initialize).
3. Type a question in the chat input:
   - *"Is bariatric surgery covered for BMI over 40?"*
   - *"What documentation is needed for gender-affirming surgery?"*
   - *"Are intrapulmonary percussive ventilation devices covered for home use?"*
4. The chatbot searches relevant policy chunks and streams an answer with citations.
5. Click **"📚 Sources"** below each answer to inspect the exact policy sections used.
6. Toggle **"🔊 Read answers aloud"** in the sidebar to hear answers via Kokoro TTS.
7. Use **"🗑️ Clear conversation"** in the sidebar to reset.

The chatbot only answers from official UHC policy documents — it explicitly says so when it lacks information rather than guessing.

---

## Why I Built It This Way

This section walks through the engineering decisions behind each component. The core challenge was: **254 UHC medical policy PDFs, each 10–80 pages, with dense clinical language, tables, and nested criteria — how do you make this searchable and answerable in real time?**

### The PDF Problem

The first thing I tried was a straightforward `pdfplumber` line-by-line extraction, and the output was terrible. UHC PDFs have multi-column layouts, repeated headers/footers on every page, table-of-contents pages with dotted leaders, and sidebar navigation text that bleeds into the main content. Some files turned out to be HTML disguised as `.pdf` extensions.

I iterated on the extractor several times. The final version (`scraper/extract_pdf_text.py`) works at the **paragraph level** instead of line level — it reconstructs paragraphs from fragmented lines, detects and removes headers/footers with regex patterns tuned to UHC's specific format, strips sidebar navigation, and skips TOC pages. It also handles tables separately and wraps them in `[TABLE]...[/TABLE]` markers so the chunker can treat them differently. HTML-disguised files get routed through BeautifulSoup instead.

I also extract structured metadata from the first few pages of each PDF — policy number, effective date, plan type (Commercial vs Medicare Advantage), and document type. This metadata becomes critical later for filtered retrieval.

### Chunking Strategy

Naive chunking (fixed 500-token windows with overlap) performs poorly on medical policy documents because a single coverage criterion can span multiple paragraphs, and you lose the logical boundary between "what's covered" and "what's not covered."

I built a **section-aware chunker** (`scraper/create_rag_chunks.py`) that first segments each document into its natural sections (Coverage Rationale, Clinical Evidence, Applicable Codes, Definitions, etc.), then applies a different chunking strategy per section type:

- **Coverage Rationale**: Split on criteria boundaries (`"The following..."`, `"For initial..."`, `"is proven and medically necessary"`). These are the most important chunks — they contain the actual yes/no coverage decisions.
- **Applicable Codes**: Table-aware chunking that keeps CPT/HCPCS code groups together and repeats the header row in each chunk.
- **Clinical Evidence**: Split on study boundaries (author-year patterns, `"A prospective..."`, professional society names). Each study stays in one chunk.
- **Everything else**: Paragraph-aware splitting with 2-sentence overlap at boundaries.

Each chunk carries rich metadata: policy name, section name, plan type, page range, and provider slug. This metadata is prepended to the chunk text before embedding, which significantly improves retrieval for queries like "bariatric surgery coverage" — the embedding now captures both the content and its source.

### Why MedEmbed

I evaluated several embedding models. General-purpose models like `all-MiniLM-L6-v2` struggle with medical terminology — they don't understand that "HFCWO" and "high-frequency chest wall oscillation" are the same thing, or that "BMI ≥ 40" is clinically relevant to "morbid obesity."

[MedEmbed-large-v0.1](https://huggingface.co/abhinand/MedEmbed-large-v0.1) is a 1024-dimensional model fine-tuned specifically for medical information retrieval. In my testing, it correctly associates paraphrased medical queries with the right policy chunks — for example, linking *"failed oral appliance therapy for sleep apnea"* to the `obstructive-sleep-apnea-treatment` policy's surgical alternatives section, even though the query never uses the word "obstructive."

The trade-off is model size (~1.3 GB) and cold-start time (~30s on HuggingFace Spaces free tier). I cache it in memory with `st.cache_resource` so subsequent queries are fast (~300ms retrieval).

### Why Qdrant

I needed a vector database that supports:
- Cosine similarity search on 1024-dim vectors
- Payload filtering (search within a specific policy, section, or provider)
- Payload indexing for fast filtered queries
- A managed cloud tier for deployment

Qdrant Cloud checked all of these. The free tier gives 1GB of storage, which is enough for ~10K chunks with metadata. I create payload indexes on `section`, `policy_name`, `plan_type`, `doc_type`, and `provider` — this lets me build features like "search only within bariatric surgery policy" or "filter to Commercial plans only" without scanning the entire collection.

Locally, I ran Qdrant via Docker during development. The same client code works for both local and cloud by switching a URL environment variable.

### The Coverage vs. Evidence Problem

This was the hardest retrieval issue I encountered. When a user asks *"Is HFCWO covered for COPD?"*, the retriever pulls chunks from both Coverage Rationale (which says "unproven and not medically necessary for COPD") and Clinical Evidence (which discusses studies about HFCWO for COPD). Without intervention, the Clinical Evidence chunks often score higher because they contain more detailed keyword overlap.

The result? The LLM would discuss the clinical studies instead of giving the definitive answer: **not covered.**

I fixed this with two mechanisms:
1. **Section priority boosting** in the retriever: Coverage Rationale chunks get a +0.04 score boost, Coverage Summary +0.03. This ensures authoritative coverage statements outrank clinical study descriptions.
2. **System prompt rule**: An explicit instruction that Coverage Rationale is the authoritative source for coverage decisions — if it says "not covered," the LLM must say so, even if Clinical Evidence discusses the treatment.

After this fix, my 100-prompt evaluation showed Section Match@1 improving from 45% to 66%, meaning the LLM sees the right section first in two-thirds of queries.

### LLM Selection: Ollama → Groq

During local development, I used **Phi-3.5 Mini** via Ollama. It's a solid small model, runs on my Mac's MPS, and is good enough for iteration. But it was slow — ~12s to first token, ~33s total for a typical answer. After tuning the prompt for brevity (2–4 bullet points max, 800-char chunk truncation), I got it down to ~3s first token and ~8s total.

For deployment, Ollama isn't viable — you can't run it on HuggingFace Spaces' free tier (2 vCPU, 16GB RAM, and the embedding model already takes 1.3GB). I switched to **Groq's API** with **Llama 3.1 8B Instant**, which gives ~560 tokens/sec inference. The `GroqClient` has the same `chat_stream()` interface as `OllamaClient`, so swapping was straightforward.

The free Groq tier has a 250K tokens-per-minute limit. I added rate-limit detection and a user-friendly error message instead of a stack trace.

### Text-to-Speech

I added optional TTS using [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) — an 82M parameter model (~300MB) that runs on CPU at near real-time speed. It auto-downloads the ONNX model files from HuggingFace Hub on first use. The toggle is off by default to avoid unnecessary compute, but it's a nice accessibility feature for medical staff who want to listen while reviewing charts.

### Prompt Engineering

The system prompt went through several iterations. Key rules that made a real difference:
- **"Answer ONLY from the policy excerpts below"** — prevents hallucination
- **"2–4 bullet points max"** — Phi-3.5 and Llama both tend to be verbose with medical content; this constraint keeps answers scannable
- **"If something is unproven and not medically necessary, say it is NOT covered"** — forces the model to make definitive statements rather than hedging
- **"Coverage Rationale is the authoritative source"** — resolves the evidence-vs-coverage confusion described above
- **Chunk deduplication** — keeps only the highest-scoring chunk per (policy, section) pair to reduce redundancy in the context window

### Deployment

I deployed on **HuggingFace Spaces** (free tier, Streamlit SDK). The main challenge was cold start: installing `torch` + `sentence-transformers` + the 1.3GB MedEmbed model download takes 2–3 minutes on first build. After that, `st.cache_resource` keeps everything in memory.

Secrets (Qdrant URL/key, Groq API key) are stored as HuggingFace Space secrets and loaded via environment variables — no credentials in code.

---

## Architecture

### High-Level Design

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Browser    │────▶│  Streamlit App (HuggingFace Spaces)          │
│   (User)     │◀────│                                              │
└─────────────┘     │  ┌─────────────┐    ┌─────────────────────┐  │
                    │  │ MedEmbed    │    │ Groq API            │  │
                    │  │ (1024-dim)  │    │ Llama 3.1 8B        │  │
                    │  │ cached RAM  │    │ 560 tok/s           │  │
                    │  └──────┬──────┘    └──────▲──────────────┘  │
                    │         │                   │                  │
                    │         ▼                   │                  │
                    │  ┌─────────────┐   context + query            │
                    │  │ Qdrant Cloud│────────────┘                 │
                    │  │ (vectors)   │                              │
                    │  └─────────────┘                              │
                    └──────────────────────────────────────────────┘
```

### Low-Level Design

#### Project Structure

```
uhc/
├── app.py                          # Streamlit web UI entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
│
├── chatbot/                        # Chatbot application layer
│   ├── config.py                   # Centralized config (LLM, retrieval, env vars)
│   ├── retriever.py                # PolicyRetriever: MedEmbed + Qdrant wrapper
│   ├── llm_groq.py                 # Groq API client (deployed)
│   ├── llm.py                      # Ollama client (local dev)
│   ├── prompts.py                  # System prompt, context formatting, deduplication
│   ├── tts.py                      # Kokoro ONNX text-to-speech
│   └── cli.py                      # CLI interface (local dev)
│
├── embedding/                      # Embedding pipeline
│   └── scripts/
│       ├── config.py               # Embedding model + Qdrant connection config
│       ├── embed_chunks.py         # Generate embeddings from RAG chunks
│       ├── store_qdrant.py         # Upsert embeddings into Qdrant with payload indexes
│       └── search.py               # Standalone search CLI for testing
│
└── scraper/                        # Data ingestion pipeline
    ├── download_policies.py        # Scrape PDFs from UHC website
    ├── extract_pdf_text.py         # PDF → structured sections with metadata
    └── create_rag_chunks.py        # Section-aware semantic chunking
```

#### Edge Cases Handled

| Edge Case | Handling |
|---|---|
| Empty / whitespace query | Warning message, no API call |
| Qdrant connection failure | Retry with exponential backoff (3 attempts) |
| Groq rate limit (429) | Caught and shown as user-friendly message |
| No relevant chunks found | "I don't have enough policy information" |
| Coverage vs. evidence conflict | System prompt + section priority boost ensures correct answer |
| Very long conversation | History trimmed to last 3 turns |
| Model loading on first visit | Spinner shown; cached with `st.cache_resource` |
| HTML files disguised as PDFs | Detected and routed through BeautifulSoup |

---

## Extending for Other Insurance Providers

The system is designed for multi-provider extensibility:

1. **Data layer**: Each chunk in Qdrant has a `provider` field (currently `"UnitedHealthcare"`). Adding a new provider means running the same pipeline with a new provider slug — chunks coexist in the same collection.
2. **Scraper**: `download_policies.py` can be adapted for any provider's website. The extractor and chunker handle standard medical policy PDF structures.
3. **Embedding**: The same MedEmbed model works for all medical content. New provider chunks are embedded and upserted alongside existing ones.
4. **Retrieval**: Add a `provider_filter` parameter to narrow results by provider, or query across all providers simultaneously.
5. **UI**: Add a provider selector dropdown in the Streamlit sidebar.

```python
retriever.retrieve(query, provider_filter="aetna")
```

---

## Local Development Setup

```bash
git clone https://github.com/mayankpatel14/uhc-policy-chatbot.git
cd uhc-policy-chatbot

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Qdrant and Groq API keys

streamlit run app.py
```

### Environment Variables

| Variable | Description | Required |
|---|---|---|
| `QDRANT_URL` | Qdrant Cloud cluster URL | Yes |
| `QDRANT_API_KEY` | Qdrant Cloud API key | Yes |
| `QDRANT_COLLECTION` | Collection name (default: `uhc_policies`) | No |
| `GROQ_API_KEY` | Groq API key ([get free](https://console.groq.com/keys)) | Yes (web) |
| `GROQ_MODEL` | Groq model (default: `llama-3.1-8b-instant`) | No |

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | [MedEmbed-large-v0.1](https://huggingface.co/abhinand/MedEmbed-large-v0.1) (1024-dim) |
| Vector Database | [Qdrant Cloud](https://qdrant.tech/) |
| LLM (deployed) | [Llama 3.1 8B](https://console.groq.com/) via Groq |
| LLM (local dev) | Phi-3.5 Mini via Ollama |
| Web Framework | Streamlit |
| Hosting | HuggingFace Spaces (free tier) |
| Text-to-Speech | [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) (82M) |
| PDF Extraction | pdfplumber + BeautifulSoup |
