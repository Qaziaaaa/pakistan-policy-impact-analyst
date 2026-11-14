# AI AGENT GROUND RULES & WORKING PROTOCOL

**Project:** Pakistan Policy Impact Analyst  
**User Role:** Senior Agentic AI Engineer  
**Workspace Boundary:** Current project directory only (Strictly no path traversal outside).

Every AI agent working in this workspace **MUST** adhere to the following rules without exception:

---

### 1. Strict File & Environment Permissions
- **No Unapproved Creation / Deletion:** Never run file/directory creation or deletion commands (`touch`, `mkdir`, `rm`, `unlink`, `truncate`, or write scripts that delete/create files or directories) without explicit user permission. Always request approval first.
- **Strict Workspace Containment:** Stay strictly inside `/home/qazi-farhan-ahmad/Desktop/New Folder`. Do not inspect, read, or modify any files outside this root unless explicitly authorized.

### 2. Git & Version Control
- **Atomic Local Commits:** Make clean, atomic local `git commit` operations after completing each validated milestone/step.
- **Zero Unprompted Pushes:** Never run `git push` or interact with remote repositories without explicit instruction.

### 3. Architecture & Clean Directory Structure
- Keep directory structures flat, intuitive, and simple.
- Use explicit, self-explanatory naming conventions (no cryptic abbreviations, no deep nested hierarchies).

### 4. Educational Methodology: Flow First, Code Second
- Do not dump code without context.
- Before introducing code for a module, provide a **simple high-level flow diagram (ASCII or Mermaid)** illustrating:
  1. Where the user request enters.
  2. How and where the LLM / Agent processes it.
  3. Where data is retrieved from, saved to (Memory/Vector DB), and monitored (LangSmith).
- Focus on architectural understanding and logic flow over verbose line-by-line syntax explanation.

### 5. API Keys & External Resources
- Never scrape, guess, or attempt to auto-discover API keys or cloud credentials.
- When an API key, endpoint, or environment variable is required (e.g., Gemini API, Qdrant Cloud, Supabase, LangSmith), **explicitly ask the user for it**.

### 6. Communication Style: Senior Engineer Brevity
- **Be Extremely Concise:** Avoid fluff, walls of text, and redundant recaps.
- Keep post-step briefings short, bulleted, and actionable so they can be read and reviewed in seconds.
- Present questions with clear, simple choices.

