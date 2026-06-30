"""
llm_file_assistant.py
======================
Part B — LLM Integration (Tool Calling / Function Calling)

Wires fs_tools.py up to Claude (Anthropic API) so natural-language queries like
  "Find resumes mentioning Python experience"
  "Create a summary file for resume_john_doe.txt"
get turned into actual tool calls, executed, and answered conversationally.

Run interactively:
    python llm_file_assistant.py

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import json
import sys
from typing import List, Dict, Any

from fs_tools import TOOL_SCHEMAS, TOOL_REGISTRY

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are a helpful file system assistant specialized in working with resume files.

You have access to 4 tools:
- read_file: read PDF/DOCX/TXT resume content
- list_files: list files in a directory, optionally filtered by extension
- write_file: write content to a file (e.g. to save a summary)
- search_in_file: search for a keyword inside a file and return matching context

Guidelines:
- When asked to act on "all resumes" or "every resume in a folder", first call list_files
  to discover what's there, then call read_file or search_in_file on each one.
- When asked to find resumes mentioning a skill/keyword, prefer search_in_file per resume
  (it's faster and gives you the matching context) over reading the whole file.
- When asked to summarize or create a new file, use write_file with clear, well-structured
  content, and tell the user the path you saved it to.
- Always explain your findings in plain language after using tools — don't just dump raw
  tool output.
- If a tool call fails, explain the error to the user in simple terms rather than retrying
  blindly forever.
"""


class LLMFileAssistant:
    """
    Orchestrates a conversation loop between the user, Claude, and fs_tools.

    Claude decides which tool(s) to call based on the user's natural language
    query; this class executes those tool calls against fs_tools.py and feeds
    the results back to Claude until it produces a final text answer.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set ANTHROPIC_API_KEY environment variable, "
                "or pass api_key= to LLMFileAssistant()."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.messages: List[Dict[str, Any]] = []

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Dispatch a tool call to the matching function in fs_tools.py."""
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**tool_input)
        except Exception as e:
            return {"success": False, "error": f"Tool execution failed: {str(e)}"}

    def ask(self, user_message: str, max_tool_rounds: int = 8, verbose: bool = True) -> str:
        """
        Send a user query, let Claude call tools as needed, and return the
        final natural-language answer.
        """
        self.messages.append({"role": "user", "content": user_message})

        for round_num in range(max_tool_rounds):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.messages
            )

            # Collect any tool_use blocks Claude wants to make
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Append assistant's turn (text + tool_use blocks) to history
            self.messages.append({"role": "assistant", "content": response.content})

            if not tool_uses:
                # No more tools needed — extract final text answer
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks).strip()

            # Execute each requested tool call, build tool_result blocks
            tool_results = []
            for tu in tool_uses:
                if verbose:
                    print(f"  🔧 Calling {tu.name}({json.dumps(tu.input)})")

                result = self._execute_tool(tu.name, tu.input)

                if verbose:
                    preview = json.dumps(result, default=str)[:150]
                    print(f"     → {preview}{'...' if len(json.dumps(result, default=str)) > 150 else ''}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, default=str)
                })

            # Feed tool results back as the next "user" turn
            self.messages.append({"role": "user", "content": tool_results})

        return "⚠️ Reached max tool-call rounds without a final answer. Try a more specific query."

    def reset(self):
        """Clear conversation history (start fresh)."""
        self.messages = []


# ── Mock mode (no API key needed) — demonstrates the tool-calling loop
#    using a simple rule-based "fake LLM" so the assistant is runnable
#    out-of-the-box for grading/demo purposes without API costs. ──────────────

class MockLLMFileAssistant:
    """
    A dependency-free stand-in for LLMFileAssistant that doesn't need an API
    key. It uses simple keyword rules to decide which fs_tools function to
    call, executes it, and formats a response — useful for offline demos and
    for verifying the tool layer works correctly before wiring up a real LLM.
    """

    def __init__(self, resume_dir: str = "resumes"):
        self.resume_dir = resume_dir

    def ask(self, user_message: str, verbose: bool = True) -> str:
        from fs_tools import list_files, read_file, search_in_file, write_file
        q = user_message.lower()

        # "Read all resumes in the X folder"
        if "read all" in q or ("read" in q and "all" in q):
            files = list_files(self.resume_dir)
            if verbose:
                print(f"  🔧 Calling list_files('{self.resume_dir}') → {len(files)} files")
            summaries = []
            for f in files:
                r = read_file(f["filepath"])
                if verbose:
                    print(f"  🔧 Calling read_file('{f['filepath']}')")
                if r["success"]:
                    name_line = r["content"].split("\n")[0]
                    summaries.append(f"- {f['filename']}: {name_line} ({r['metadata']['word_count']} words)")
            return f"Read {len(summaries)} resumes:\n" + "\n".join(summaries)

        # "Find resumes mentioning X"
        if "find" in q or "mention" in q or "search" in q:
            keyword = self._extract_keyword(user_message)
            files = list_files(self.resume_dir)
            if verbose:
                print(f"  🔧 Calling list_files('{self.resume_dir}')")
            hits = []
            for f in files:
                r = search_in_file(f["filepath"], keyword)
                if verbose:
                    print(f"  🔧 Calling search_in_file('{f['filepath']}', '{keyword}')")
                if r["success"] and r["match_count"] > 0:
                    hits.append(f"- {f['filename']} ({r['match_count']} match{'es' if r['match_count'] > 1 else ''}): {r['matches'][0]['context']}")
            if not hits:
                return f"No resumes found mentioning '{keyword}'."
            return f"Found {len(hits)} resume(s) mentioning '{keyword}':\n" + "\n".join(hits)

        # "Create a summary file for X"
        if "summary" in q or "summarize" in q:
            filename = self._extract_filename(user_message, self.resume_dir)
            if not filename:
                return "Please specify which resume file to summarize."
            filepath = os.path.join(self.resume_dir, filename)
            r = read_file(filepath)
            if verbose:
                print(f"  🔧 Calling read_file('{filepath}')")
            if not r["success"]:
                return f"Couldn't read {filename}: {r['error']}"

            content = r["content"]
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            summary = "\n".join(lines[:8])  # naive summary: first 8 non-empty lines
            out_path = os.path.join("reports", f"summary_{os.path.splitext(filename)[0]}.txt")
            w = write_file(out_path, f"SUMMARY OF {filename}\n{'='*40}\n{summary}\n")
            if verbose:
                print(f"  🔧 Calling write_file('{out_path}', ...)")
            if w["success"]:
                return f"Summary saved to {out_path} ({w['bytes_written']} bytes written)."
            return f"Failed to save summary: {w['error']}"

        # "List files"
        if "list" in q:
            files = list_files(self.resume_dir)
            if verbose:
                print(f"  🔧 Calling list_files('{self.resume_dir}')")
            return f"Found {len(files)} files:\n" + "\n".join(f"- {f['filename']}" for f in files)

        return "I can help you read, search, list, or summarize resumes. Try: 'Find resumes mentioning Python' or 'Read all resumes in the resumes folder'."

    def _extract_keyword(self, text: str) -> str:
        import re
        m = re.search(r'mentioning\s+(\w+)', text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r'with\s+(\w+)\s+experience', text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r'(?:find|search)(?:\s+for)?\s+(\w+)', text, re.IGNORECASE)
        if m:
            return m.group(1)
        return "Python"

    def _extract_filename(self, text: str, directory: str) -> str:
        files = os.listdir(directory) if os.path.exists(directory) else []
        for f in files:
            base = os.path.splitext(f)[0]
            if base.lower() in text.lower() or f.lower() in text.lower():
                return f
        return ""


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🤖 LLM File System Assistant")
    print("=" * 60)

    use_mock = False
    if not ANTHROPIC_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  No ANTHROPIC_API_KEY found (or anthropic package missing).")
        print("    Falling back to MOCK mode — demonstrates the same tool-calling")
        print("    flow using rule-based query routing instead of a real LLM.")
        print("    Set ANTHROPIC_API_KEY to use real Claude tool calling.\n")
        assistant = MockLLMFileAssistant(resume_dir="resumes")
        use_mock = True
    else:
        assistant = LLMFileAssistant()
        print("\n✅ Connected to Claude — full LLM tool calling enabled.\n")

    print("Try queries like:")
    print('  - "Read all resumes in the resumes folder"')
    print('  - "Find resumes mentioning Python experience"')
    print('  - "Create a summary file for resume_john_doe"')
    print('  - "List files in resumes"')
    print("  - 'quit' to exit\n")

    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        answer = assistant.ask(query)
        print(f"\n🤖 Assistant:\n{answer}\n")


if __name__ == "__main__":
    main()
