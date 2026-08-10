"""
article_pipeline.py

A multi-agent LangGraph pipeline, now with a human-in-the-loop review step
before anything is saved:

    RESEARCH -> WRITE -> VERIFY --[not verified, attempts left]--> RESEARCH (loop)
                                 --[verified, or attempts exhausted]--> END

The graph produces a DRAFT and STOPS — it no longer saves automatically.
From there, the caller (server.py) can:
    - call revise_draft() any number of times, based on user feedback
    - call verify_draft() again to refresh the verified/unverified badge
      after an edit
    - call save_article() only once the user explicitly confirms

This mirrors a real editorial workflow: research + draft + auto-check happens
first, but a human always gets the final say before anything is written to
disk.

Requires:
    pip install langgraph langchain-tavily
    TAVILY_API_KEY set in your .env
"""

import os
import re
from datetime import datetime
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

ARTICLES_DIR = "data/articles"
MAX_ATTEMPTS = 3


class ArticleState(TypedDict):
    topic: str
    user_email: str
    search_query: str            # may get refined between attempts
    sources: List[dict]          # [{title, url, content}, ...]
    draft: str
    verified: bool
    issues: str                  # why verification failed, used to refine next search
    attempts: int


def get_user_articles_dir(email: str) -> str:
    safe_id = email.lower().replace("@", "_at_").replace(".", "_")
    user_dir = os.path.join(ARTICLES_DIR, safe_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    return re.sub(r"[\s-]+", "-", text)[:60]


# ---------------- standalone helpers, usable both inside the graph and
# ---------------- independently by server.py during the review step ----------------

def verify_draft(llm, draft: str, sources: List[dict]) -> dict:
    """Checks a draft against its sources. Returns {"verified": bool, "issues": str}.
    Used both by the graph's verify node, and again after a manual revision so
    the badge stays accurate."""
    sources_text = "\n\n".join(f"Source: {s['title']}\n{s['content']}" for s in sources)

    verify_prompt = f"""Sources:
{sources_text}

Draft article:
{draft}

Does the draft rely only on facts present in the sources above?
Answer in EXACTLY this format, two lines, nothing else:
VERIFIED: yes
ISSUES: none

or

VERIFIED: no
ISSUES: <short description of the unsupported claim or missing coverage>"""

    raw = llm.invoke(verify_prompt).content.strip()
    print(f"[verify_draft] raw model response:\n{raw}\n")  # temporary debug logging

    verified = False
    issues = "Could not confidently parse verification response."

    match = re.search(r"VERIFIED:\s*(yes|no)", raw, re.IGNORECASE)
    if match:
        verified = match.group(1).lower() == "yes"
        issues_match = re.search(r"ISSUES:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
        issues = issues_match.group(1).strip() if issues_match else ""
        if verified:
            issues = ""

    return {"verified": verified, "issues": issues}


def serialize_document(title: str, status_text: str, draft: str, sources: List[dict]) -> str:
    """Turns title + status line + body + sources into one plain-text
    document the LLM can see and edit as a whole — this is what makes
    EVERYTHING in the letter editable through a single instruction, not
    just the body text."""
    if sources:
        sources_lines = "\n".join(f"- {s['title']} | {s['url']}" for s in sources)
    else:
        sources_lines = "- (none)"
    status_display = status_text if status_text else "(none)"
    return f"TITLE: {title}\n\nSTATUS: {status_display}\n\nBODY:\n{draft}\n\nSOURCES:\n{sources_lines}"


def parse_document(text: str, original_sources: List[dict] = None) -> dict:
    """Parses the LLM's revised document back into
    {title, status_text, draft, sources}. For any source whose title+url
    still matches one we already had, its original full source `content`
    (used for grounding/verification) is carried over — the LLM never has
    to retype full source content, it only edits the short title/url line."""
    original_sources = original_sources or []
    content_lookup = {(s["title"], s["url"]): s.get("content", "") for s in original_sources}

    title_match = re.search(r"TITLE:\s*(.*)", text)
    status_match = re.search(r"STATUS:\s*(.*?)\n\nBODY:", text, re.DOTALL)
    body_match = re.search(r"BODY:\s*\n(.*?)\nSOURCES:", text, re.DOTALL)
    sources_match = re.search(r"SOURCES:\s*\n(.*)", text, re.DOTALL)

    title = title_match.group(1).strip() if title_match else ""

    status_text = status_match.group(1).strip() if status_match else ""
    if status_text.lower() in ("(none)", "none", ""):
        status_text = ""

    draft = body_match.group(1).strip() if body_match else text.strip()

    sources = []
    if sources_match:
        for line in sources_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("-").strip()
            if not line or line.lower() == "(none)":
                continue
            if "|" in line:
                t, u = line.split("|", 1)
                t, u = t.strip(), u.strip()
            else:
                t, u = line, ""
            sources.append({"title": t, "url": u, "content": content_lookup.get((t, u), "")})

    return {"title": title, "status_text": status_text, "draft": draft, "sources": sources}


def revise_document(llm, title: str, status_text: str, draft: str, sources: List[dict], instruction: str) -> dict:
    """Applies a plain-language instruction to the WHOLE letter — title,
    status line, body, or the sources list — not just the body text.
    Returns the parsed {title, status_text, draft, sources} after the edit."""
    current_doc = serialize_document(title, status_text, draft, sources)

    revise_prompt = f"""Current document:
{current_doc}

The user requested this change: "{instruction}"

Apply the change wherever it belongs — the title, the status line, the body
text, or the sources list (you can add, remove, or edit any of them,
including removing the status line entirely by writing "STATUS: (none)").
Leave everything else unchanged unless the instruction implies otherwise.
Don't invent new source URLs — only add/keep/remove source lines already
present unless the user explicitly gives you a new one to add.

Respond with ONLY the full revised document, in EXACTLY this format, nothing before or after it:
TITLE: <title>

STATUS: <status line, or "(none)" if it should be removed>

BODY:
<full article body>

SOURCES:
- <source title> | <url>
- <source title> | <url>
(or "- (none)" if there are no sources)"""

    raw = llm.invoke(revise_prompt).content
    return parse_document(raw, original_sources=sources)


def default_status_text(verified: bool, attempts: int) -> str:
    """The initial status line text, generated once right after verification
    completes. After this, it becomes just another editable part of the
    document — the user can change or remove it, and it will stay that way
    on future revisions instead of being silently regenerated."""
    if verified:
        return "Status: VERIFIED (confirmed against sources)"
    return f"Status: UNVERIFIED (after {attempts} attempt(s) — treat with caution)"


def build_preview(title: str, status_text: str, draft: str, sources: List[dict]) -> str:
    """Builds the markdown preview shown to the user (and, later, saved to
    disk unchanged) — used identically for both the pending-review preview
    and the final saved file, so what you approve is exactly what gets
    written.

    `status_text` is used exactly as given — including empty, if the user
    asked to remove it — rather than being recomputed from verified/attempts
    every time. That recomputation was the bug: any edit or removal request
    to the status line was being silently overwritten on the next preview
    build. Same fix pattern as the Sources section.

    If there are no sources left (e.g. the user asked to remove them all),
    the Sources section is omitted entirely rather than showing a
    placeholder line — the letter should read as a clean, ready-to-post
    piece of writing, not an internal debug artifact."""
    header = f"# {title}\n\n"
    if status_text:
        header += f"*{status_text}*\n\n"

    if sources:
        sources_list = "\n".join(f"- [{s['title']}]({s['url']})" for s in sources)
        return f"{header}{draft}\n\n## Sources\n{sources_list}\n"

    return f"{header}{draft}\n"


def save_article(topic: str, title: str, status_text: str, verified: bool, user_email: str, draft: str, sources: List[dict]) -> dict:
    """Writes the final, user-approved article to disk exactly as previewed.
    `topic` (the original search topic) is used only for the filename slug
    and stays stable even if the title was edited; `title`/`status_text` are
    whatever the user last approved, verbatim.

    A hidden HTML comment with the TRUE verified state is written on its own
    line at the top of the file — invisible when rendered as markdown, but
    readable by list_my_articles() in server.py. This keeps the history
    list's badge accurate even after the user edits or deletes the visible
    status_text line, since that text is no longer a reliable signal once
    it's freely user-editable."""
    content = build_preview(title, status_text, draft, sources)
    meta_comment = f"<!-- verified:{'true' if verified else 'false'} -->\n"
    full_content = meta_comment + content

    user_dir = get_user_articles_dir(user_email)
    filename = f"{slugify(topic)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    path = os.path.join(user_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(full_content)

    return {
        "final_article": content,  # returned WITHOUT the hidden comment, for display
        "saved_path": path,
        "filename": filename,
    }


# ---------------- the graph itself: research -> write -> verify (loop) -> END ----------------

def build_article_graph(llm):
    """Builds and compiles the research/write/verify pipeline. Stops after
    verification — does NOT save. `llm` is the same ChatOpenAI object
    already built in server.py."""

    search_tool = TavilySearch(max_results=5)

    def research_node(state: ArticleState) -> dict:
        query = state.get("search_query") or state["topic"]
        try:
            result = search_tool.invoke({"query": query})
            raw_results = result.get("results", []) if isinstance(result, dict) else []
        except Exception as e:
            raw_results = []
            print(f"Tavily search failed: {e}")

        sources = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in raw_results
        ]
        return {"sources": sources, "attempts": state.get("attempts", 0) + 1}

    def write_node(state: ArticleState) -> dict:
        sources_text = "\n\n".join(
            f"Source: {s['title']} ({s['url']})\n{s['content']}" for s in state["sources"]
        )
        if not sources_text.strip():
            return {"draft": f"No source material could be found on '{state['topic']}'."}

        write_prompt = f"""Write a clear, well-structured article about: {state['topic']}

Use ONLY the information in the sources below. Do not add facts that aren't supported by them.

{sources_text}

Write the article now (a few paragraphs, plain text, no preamble):"""

        draft = llm.invoke(write_prompt).content
        return {"draft": draft}

    def verify_node(state: ArticleState) -> dict:
        return verify_draft(llm, state["draft"], state["sources"])

    def prepare_retry_node(state: ArticleState) -> dict:
        refined_query = f"{state['topic']} {state['issues']}".strip()
        return {"search_query": refined_query}

    def decide_after_verify(state: ArticleState) -> str:
        if state["verified"]:
            return "done"
        if state["attempts"] < MAX_ATTEMPTS:
            return "retry"
        return "done"  # attempts exhausted — stop anyway, flagged unverified

    graph = StateGraph(ArticleState)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("verify", verify_node)
    graph.add_node("prepare_retry", prepare_retry_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "verify")
    graph.add_conditional_edges("verify", decide_after_verify, {
        "done": END,
        "retry": "prepare_retry",
    })
    graph.add_edge("prepare_retry", "research")

    return graph.compile()