
# Ask Abdelrahman: a RAG assistant for my portfolio

A question-answering assistant that answers questions about my projects, skills and experience, **using only my own documents**. Every answer shows the sources it came from, so you can check it.


**Author:** Abdelrahman Gaber · [GitHub](https://github.com/AbdelrahmanGaber528)

---

## What it does

- Answers questions such as *"Does he have cloud experience?"* or *"What is SEMSOL?"* in natural language.
- Shows the **source records** behind each answer (title, type, text used).
- Says *"I don't know"* when the answer is not in the data, instead of inventing one.
- Does not claim sole authorship of team or third-party projects (each project has an `ownership` field).

## How it works

```
README files + LinkedIn + notes
        │  (cleaned, split by meaning, reviewed by hand)
        ▼
 knowledge.json  ──►  Documents  ──►  Embeddings  ──►  FAISS vector store
                                                              │
 question ──► retrieve top-k ──► add parent project ──► prompt ──► LLM ──► answer + sources
```

1. **Loading:** `knowledge.json` is read into LangChain `Document` objects, with metadata (`id`, `type`, `title`, `parent_id`, `ownership`).
2. **Chunking:** none at query time. The data is already split so that **one record = one idea** (about 40 to 130 words), so a splitter would only cut records in half.
3. **Embedding:** each record is embedded as a short header (type, title, tags) plus its text, using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieval:** FAISS similarity search returns the top-k records. Each detail section also brings in its **parent project** record, so answers have the overview and the detail.
5. **Generation:** a system prompt holds the rules, and a human message holds the retrieved context and the question.

## Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.10+ |
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (CPU) |
| Vector store | FAISS |
| LLM | OpenRouter (OpenAI-compatible API) |
| UI | Gradio |
| Hosting | Hugging Face Spaces |

## Project structure

```
.
├── LICENSE
├── knowledge.json       
├── requirements.txt
├── README.md
└── self-chat-rag.ipynb 
```

## The data format

`knowledge.json` is a list of records:

```json
{
  "id": "proj-ocr",
  "type": "project",
  "title": "Distributed OCR System",
  "text": "Distributed OCR System is a microservices-based ...",
  "source": "github",
  "tags": ["ocr", "fastapi", "aws"],
  "ownership": "team",
  "parent_id": null
}
```

| Field | Purpose |
|---|---|
| `id` | Unique name, used for citations and parent lookup |
| `type` | `about`, `experience`, `education`, `project`, `repo_section`, `skill`, `goal` |
| `text` | The content that is embedded and searched |
| `tags` | Keywords added to the embedded header |
| `ownership` | `own`, `team`, `unknown` or `third_party`, so the assistant does not overclaim |
| `parent_id` | Links a detail section to its project |
| `evidence` | On skills: the records that prove the skill |

Private details (email, phone, credentials) are intentionally **not** included.

## Run it locally

```bash
git clone https://github.com/AbdelrahmanGaber528/Cellula_3Week_Abdelrahman_Gaber.git
cd Cellula_3Week_Abdelrahman_Gaber

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export OPENROUTER_API_KEY="your-key"
```


## `requirements.txt`

```
gradio
langchain-core
langchain-openai
langchain-community
langchain-classic
sentence-transformers
faiss-cpu
```

## Evaluation

Retrieval is measured with **Recall@K** on a hand-written question set . Each question lists the record ids that should be found.

| Setup | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| Embedded text only | _fill in_ | _fill in_ | _fill in_ |
| Header + text (current) | _fill in_ | _fill in_ | _fill in_ |
| Header + text + parent expansion | _fill in_ | _fill in_ | _fill in_ |

_Run the notebook and replace the placeholders with your own numbers._

## License

The MIT License (MIT)

Copyright (c) 2011-2026 The Bootstrap Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.