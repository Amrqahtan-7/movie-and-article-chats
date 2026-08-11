"""
rag_graph.py

A LangGraph pipeline for handling multi-step chat questions:

    ROUTE -> RETRIEVE (branches by mode) -> ANSWER -> VERIFY -> LABEL

Modes:
    - "compare"    : question compares two of the user's own uploaded files
    - "user_files" : question is about the user's own uploaded file(s)
    - "dataset"    : question is about the base movie dataset
    - "general"    : general knowledge question, no retrieval needed

This file expects `vectorstore`, `llm`, and `question_answer_chain` to already
exist (built in server.py). Import this module AFTER those are created, and
pass them in via build_graph(...).
"""

import os
import json
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from qdrant_client.http import models as qdrant_models


class ChatState(TypedDict):
    question: str
    user_email: str
    user_files: List[str]       # this user's own uploaded filenames
    mode: str                   # "compare" | "user_files" | "dataset" | "general"
    file_a: Optional[str]
    file_b: Optional[str]
    context: List[Document]
    context_a: List[Document]
    context_b: List[Document]
    draft_answer: str
    final_answer: str
    source_label: str


def build_graph(vectorstore, llm, question_answer_chain, get_user_upload_dir):
    """Builds and compiles the graph. Call this once at server startup,
    passing in the objects already created in server.py."""

    # ---------------- helpers ----------------

    def search_scoped(query: str, owner_filter, k: int = 4):
        return vectorstore.similarity_search(query, k=k, filter=owner_filter)

    def owner_filter_for(email: str):
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.uploaded_by",
                    match=qdrant_models.MatchValue(value=email),
                )
            ]
        )

    def file_filter_for(email: str, filename: str):
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.uploaded_by",
                    match=qdrant_models.MatchValue(value=email),
                ),
                qdrant_models.FieldCondition(
                    key="metadata.title",
                    match=qdrant_models.MatchValue(value=filename),
                ),
            ]
        )

    def dataset_filter():
        return qdrant_models.Filter(
            must=[
                qdrant_models.IsNullCondition(
                    is_null=qdrant_models.PayloadField(key="metadata.uploaded_by")
                )
            ]
        )

    # ---------------- nodes ----------------

    def route_node(state: ChatState) -> dict:
        user_dir = get_user_upload_dir(state["user_email"])
        user_files = os.listdir(user_dir)

        # Ask the LLM to classify the question rather than relying on brittle
        # keyword matching — this handles "compare X and Y" as well as more
        # natural phrasing without hardcoded rules.
        classify_prompt = f"""You are a routing classifier for a chatbot. The user has these uploaded files: {user_files}

Question: "{state['question']}"

Decide the mode:
- "compare": the question compares two of the listed files. Also return file_a and file_b as exact filenames from the list above.
- "user_files": the question is about one or more of the listed files, but not a comparison.
- "dataset": the question is about a movie plot/dataset, not the user's own files.
- "general": general knowledge, greetings, small talk, or anything unrelated to any file or the movie dataset.

If the question is short, vague, a greeting (like "hello", "hi", "how are you"), or doesn't clearly reference a file or movie, choose "general" — do NOT guess "user_files" just because the user happens to have files uploaded.

Choose "user_files" ONLY if the question actually points at the user's own documents: it names one of the files above, says "my file/document/report/upload", or asks about something the user has already said is in them. A bare topic word that merely sounds like it could turn up in a business document — "portfolio", "revenue", "strategy", "the team" — is NOT enough on its own. Test it this way: if the question would make perfect sense asked to a chatbot with no uploaded files at all, it is "general".

Respond with ONLY compact JSON, no other text:
{{"mode": "...", "file_a": "..." or null, "file_b": "..." or null}}"""

        raw = llm.invoke(classify_prompt).content.strip()
        try:
            # strip accidental markdown fences if the model adds them
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
        except Exception:
            # Safe fallback: if classification fails, default to "general"
            # rather than guessing "user_files". Guessing user_files just
            # because the account HAS files (regardless of what the
            # question actually was) caused greetings like "hello" to
            # wrongly pull back file content. Defaulting to general means
            # a misclassification produces a harmless plain-chat answer,
            # not an unwanted file retrieval.
            parsed = {"mode": "general", "file_a": None, "file_b": None}

        return {
            "user_files": user_files,
            "mode": parsed.get("mode", "dataset"),
            "file_a": parsed.get("file_a"),
            "file_b": parsed.get("file_b"),
        }

    def retrieve_compare_node(state: ChatState) -> dict:
        email = state["user_email"]
        ctx_a = search_scoped(state["question"], file_filter_for(email, state["file_a"])) if state["file_a"] else []
        ctx_b = search_scoped(state["question"], file_filter_for(email, state["file_b"])) if state["file_b"] else []
        return {"context_a": ctx_a, "context_b": ctx_b, "context": ctx_a + ctx_b}

    def retrieve_user_files_node(state: ChatState) -> dict:
        docs = search_scoped(state["question"], owner_filter_for(state["user_email"]))
        return {"context": docs}

    def retrieve_dataset_node(state: ChatState) -> dict:
        docs = search_scoped(state["question"], dataset_filter())
        return {"context": docs}

    def retrieve_general_node(state: ChatState) -> dict:
        return {"context": []}

    def route_to_retrieve(state: ChatState) -> str:
        return {
            "compare": "retrieve_compare",
            "user_files": "retrieve_user_files",
            "dataset": "retrieve_dataset",
            "general": "retrieve_general",
        }.get(state["mode"], "retrieve_dataset")

    def answer_node(state: ChatState) -> dict:
        if state["mode"] == "general" or not state["context"]:
            if state["mode"] == "general":
                draft = llm.invoke(state["question"]).content
            else:
                draft = "I don't have relevant information in the available files or dataset to answer that."
            return {"draft_answer": draft}

        result = question_answer_chain.invoke({"input": state["question"], "context": state["context"]})
        draft = result if isinstance(result, str) else result.get("answer", str(result))
        return {"draft_answer": draft}

    def verify_node(state: ChatState) -> dict:
        if state["mode"] == "general" or not state["context"]:
            return {"final_answer": state["draft_answer"]}

        context_text = "\n---\n".join(d.page_content for d in state["context"])
        check_prompt = f"""Context:
{context_text}

Draft answer: "{state['draft_answer']}"

Does the draft answer rely only on facts present in the context above? If yes, repeat it unchanged. If it includes claims not supported by the context, rewrite it to remove or flag those unsupported parts. Respond with ONLY the final answer text, no explanation."""

        verified = llm.invoke(check_prompt).content.strip()
        return {"final_answer": verified}

    def label_node(state: ChatState) -> dict:
        labels = {
            "compare": f"[Source: your files — {state.get('file_a') or '?'} vs {state.get('file_b') or '?'}]",
            "user_files": "[Source: your uploaded file(s)]",
            "dataset": "[Source: movie dataset]",
            "general": "[Source: general knowledge — not from your files or the dataset]",
        }
        label = labels.get(state["mode"], "")
        return {"final_answer": f"{state['final_answer']}\n\n{label}"}

    # ---------------- graph wiring ----------------

    graph = StateGraph(ChatState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve_compare", retrieve_compare_node)
    graph.add_node("retrieve_user_files", retrieve_user_files_node)
    graph.add_node("retrieve_dataset", retrieve_dataset_node)
    graph.add_node("retrieve_general", retrieve_general_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)
    graph.add_node("label", label_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges("route", route_to_retrieve, {
        "retrieve_compare": "retrieve_compare",
        "retrieve_user_files": "retrieve_user_files",
        "retrieve_dataset": "retrieve_dataset",
        "retrieve_general": "retrieve_general",
    })
    graph.add_edge("retrieve_compare", "answer")
    graph.add_edge("retrieve_user_files", "answer")
    graph.add_edge("retrieve_dataset", "answer")
    graph.add_edge("retrieve_general", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", "label")
    graph.add_edge("label", END)

    return graph.compile()