# Lullus — Local AI Knowledge Organizer & Smart Notes

> A privacy-first, locally-running AI assistant for organizing knowledge, retrieving precise information from complex texts, and transforming rough notes into polished documents. Powered by Ollama — **your data never leaves your computer**.

---

## What Lullus Does

Lullus is built for anyone who works with dense, complex texts — academic papers, textbooks, technical documentation, legal texts, lecture notes. You upload your materials, and Lullus indexes them into a searchable knowledge base. Then you can:

- **Ask questions** and get answers grounded exclusively in your uploaded materials, with source citations
- **Choose how you want answers**: precise and short, or comprehensive and detailed
- **Transform rough notes** into polished, structured documents enriched with content from your knowledge base
- **Generate exercises** to test your understanding of the material
- **Run adaptive assessments** that adjust difficulty based on your performance

Lullus is not a generic chatbot. It answers **only from your materials** and explicitly says "I don't know" when it can't find relevant information. No hallucinations, no made-up references.

---

## Key Features

### Knowledge Base with Discipline-Specific Chunking
Upload PDFs, DOCX, PPTX, EPUB, HTML, TXT, Markdown, and CSV files. When uploading, choose how the text should be split for optimal retrieval:

| Strategy | Best For | Chunk Size | Behavior |
|----------|----------|------------|----------|
| **Humanities / Social Sciences** | Papers, essays, philosophy, law, history | 1200 chars | Preserves paragraph boundaries, argumentative flow, block quotes, discourse markers |
| **STEM / Technical** | Math, physics, CS, engineering, medicine | 384 chars | Equation-aware, preserves definitions, theorems, proofs as atomic units |
| **Auto (Balanced)** | Mixed content, general notes | 512 chars | Good default for most documents |

### Two Retrieval Modes
Switch between answer styles depending on what you need:

- **Precise** — Short, direct, factual answers. Gets to the point. Best for quick lookups and fact-checking.
- **Exhaustive** — Long, comprehensive, deeply detailed answers. Covers every angle found in the materials. Best for deep study and review.

Both modes answer **only from your knowledge base** and say "I don't have information on this in your materials" when nothing relevant is found.

### Smart Notes
Paste rough notes, keywords, bullet points, or half-finished sentences. Lullus transforms them into polished, well-structured text by:
- Elaborating every point into full explanations
- Integrating relevant information from your knowledge base
- Filling in gaps and adding context
- Citing source materials

Output styles: Detailed, Concise, or Outline.

### Knowledge Assessment
- **Practice Exercises** — Multiple choice, open-ended, fill-in-the-blank, true/false, problem-solving, and code exercises, all generated from your materials
- **Adaptive Assessment** — Questions that adjust difficulty based on your performance, with a final report showing strengths and gaps
- **History** — Track past assessments over time

### Chat
Conversational interface with quick-action buttons, source citations with confidence scores, and conversation export.

### Fully Local & Private
Everything runs on your machine via Ollama. No cloud, no API keys, no data sharing. The only exception is the optional DuckDuckGo web search.

---

## Quick Start

### 1. Install Ollama

**macOS:** `brew install ollama`
**Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
**Windows:** Download from [ollama.com/download](https://ollama.com/download)

### 2. Pull the models

```bash
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull nomic-embed-text
```

### 3. Setup

```bash
cd unimentor
pip install -r requirements.txt
```

### 4. Launch

```bash
streamlit run app/main.py
```

Open `http://localhost:8501`.

---

## System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Disk** | 10 GB | 20 GB |
| **Python** | 3.10+ | 3.12+ |
| **GPU** | Not required | NVIDIA 6GB+ VRAM |

---

## Configuration

Edit `config/user_config.yaml`:

```yaml
llm:
  model: "mistral:7b-instruct-v0.3-q4_K_M"
  temperature: 0.3
  max_tokens: 2048

embeddings:
  chunk_size: 512
  chunk_overlap: 50

rag:
  top_k: 5
  similarity_threshold: 0.3
```

Or change settings from the Settings page in the UI.

---

## Supported LLMs

Any Ollama model works. Recommended:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| **Mistral 7B** (default) | 4.1 GB | Fast | Good balance |
| **Phi-3 Mini** | 2.3 GB | Fastest | Good for low-resource machines |
| **Llama 3.1 8B** | 4.7 GB | Medium | Higher quality |
| **Gemma 2 9B** | 5.4 GB | Medium | Strong multilingual |

---

## FAQ

**Does it need internet?**
No. Everything is local. Web search is optional.

**Is my data private?**
Yes. Nothing leaves your machine. No telemetry, no analytics, no cloud calls.

**What if the answer isn't in my materials?**
Lullus will tell you: "I don't have information on this in your materials." It does not make things up.

**Can I use it without a GPU?**
Yes. Quantized models (Q4_K_M) run well on CPU. Expect 5-15s per response with 8GB RAM.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Lullus</strong> — Organize knowledge. Retrieve precisely. Write clearly.
</p>
