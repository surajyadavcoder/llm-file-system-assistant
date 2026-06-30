#  LLM-Powered File System Assistant


---

##  Overview

This project has two parts:

- **Part A** (`fs_tools.py`) — Four structured, LLM-callable file system tools: `read_file`, `list_files`, `write_file`, `search_in_file`. Each returns a predictable dict/list, never raises, and supports PDF, DOCX, and TXT resumes.
- **Part B** (`llm_file_assistant.py`) — Wires those tools to **Claude (Anthropic API)** via function calling, so natural-language queries like *"Find resumes mentioning Python experience"* automatically trigger the right tool calls.

A dependency-free **Mock mode** is included too, so the assistant is runnable and demoable even without an API key — useful for grading/testing the tool layer in isolation.

---

##  Project Structure

```
llm_file_assistant/
├── fs_tools.py              ← Part A: 4 core file system tools
├── llm_file_assistant.py    ← Part B: LLM integration + CLI
├── requirements.txt
├── README.md
├── resumes/                 ← 10 sample resumes (8 TXT, 1 PDF, 1 DOCX)
└── reports/                 ← Output: generated summary files
```

---

##  Part A — Core File System Tools (`fs_tools.py`)

### `read_file(filepath: str) → dict`
Reads PDF (`pypdf`), DOCX (`python-docx`), or TXT/MD content and returns:
```python
{
    "success": True,
    "content": "...",
    "metadata": {"filename": "...", "extension": ".pdf", "size_bytes": 2048,
                 "char_count": 1500, "word_count": 240},
    "error": None
}
```
Gracefully returns `success: False` with a clear `error` message for missing files, unsupported types, or parse failures — never raises an exception, so an LLM orchestrator can always branch safely.

### `list_files(directory: str, extension: str = None) → list`
Returns file metadata (name, path, extension, size, modified date), optionally filtered by extension (`.pdf`, `.txt`, etc — leading dot optional, case-insensitive).

### `write_file(filepath: str, content: str) → dict`
Writes content to disk, auto-creating any missing parent directories. Returns success status and bytes written.

### `search_in_file(filepath: str, keyword: str) → dict`
Case-insensitive keyword search inside a file (reuses `read_file` internally, so it works across PDF/DOCX/TXT). Returns every match with ~80 characters of surrounding context.

### Tool Schemas
`fs_tools.py` also exports `TOOL_SCHEMAS` (Anthropic-format JSON schemas) and `TOOL_REGISTRY` (name → function map), so Part B can plug them directly into the Claude API.

---

##  Part B — LLM Integration (`llm_file_assistant.py`)

### How it works
1. User sends a natural language query.
2. Claude receives the query + `TOOL_SCHEMAS` and decides which tool(s) to call.
3. `LLMFileAssistant._execute_tool()` dispatches the call to the real Python function in `fs_tools.py`.
4. The tool's JSON result is sent back to Claude as a `tool_result`.
5. Claude either calls more tools (e.g. one `search_in_file` per resume) or returns a final natural-language answer.

```python
from llm_file_assistant import LLMFileAssistant

assistant = LLMFileAssistant()  # reads ANTHROPIC_API_KEY from env
answer = assistant.ask("Find resumes mentioning Python experience")
print(answer)
```

### Example Queries (from the assignment brief)
| Query | What happens |
|---|---|
| "Read all resumes in the resumes folder" | `list_files` → `read_file` × N |
| "Find resumes mentioning Python experience" | `list_files` → `search_in_file` × N |
| "Create a summary file for resume_john_doe" | `read_file` → `write_file` |

### Mock Mode (no API key required)
If `ANTHROPIC_API_KEY` isn't set, `llm_file_assistant.py` automatically falls back to `MockLLMFileAssistant` — a tiny rule-based router that calls the exact same `fs_tools.py` functions, so you can verify and demo the tool layer without any API cost or key setup.

---

##  Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your Anthropic API key for real LLM tool calling
```bash
export ANTHROPIC_API_KEY=sk-ant-...        # macOS/Linux
setx ANTHROPIC_API_KEY "sk-ant-..."        # Windows
```
Without this, the assistant runs in Mock mode automatically — no setup needed.

### 3. Run the self-test for the tools
```bash
python fs_tools.py
```

### 4. Run the interactive assistant
```bash
python llm_file_assistant.py
```
Then try:
```
You > Find resumes mentioning Python experience
You > Read all resumes in the resumes folder
You > Create a summary file for resume_john_doe
You > List files in resumes
```

---

##  Sample Data

10 resumes covering different roles and all 3 supported formats:

| File | Format | Role |
|---|---|---|
| resume_john_doe.txt | TXT | Backend Developer |
| resume_priya_sharma.txt | TXT | Frontend Developer |
| resume_arjun_mehta.txt | TXT | Data Scientist |
| resume_sara_khan.txt | TXT | DevOps Engineer |
| resume_vikram_singh.txt | TXT | Full Stack Developer |
| resume_neha_gupta.txt | TXT | ML Engineer |
| resume_rohan_kapoor.txt | TXT | QA Engineer |
| resume_ananya_iyer.txt | TXT | Product Manager |
| resume_kavya_nair.pdf | **PDF** | Cloud Engineer |
| resume_aditya_rao.docx | **DOCX** | Site Reliability Engineer |

---

##  Production Notes

- `search_in_file` uses simple regex matching — for fuzzy/semantic search, pair this with the RAG pipeline from Milestone 2.
- `max_tool_rounds` in `LLMFileAssistant.ask()` caps tool-calling loops at 8 rounds to prevent runaway costs on malformed queries.
- For very large resume folders, consider batching `read_file` calls or adding pagination to `list_files`.

---

##  Author

**Suraj Yadav** | GitHub: https://github.com/surajyadavcoder | Email: Surajyadavx.in@gmail.com
