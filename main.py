"""
main.py

FastAPI service exposing:
  GET  /health  — readiness check
  POST /chat    — stateless conversational SHL assessment recommender

Architecture:
  1. Receive full conversation history (stateless)
  2. Build a retrieval query from the history
  3. Semantic search over SHL catalog (FAISS)
  4. Inject catalog excerpts + history into LLM prompt
  5. Parse and validate LLM response
  6. Return structured JSON
"""

import json
import logging
import os
import re
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from prompts import (
    SYSTEM_PROMPT,
    build_retrieval_query,
    build_user_prompt,
)
from retriever import get_retriever

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MAX_TURNS = 8          # Hard cap from assignment
TOP_K_RETRIEVAL = 15   # Retrieve more, let LLM pick best

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI that recommends SHL assessments for hiring needs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages list cannot be empty")
        if len(v) > MAX_TURNS * 2:
            raise ValueError(f"Conversation exceeds maximum {MAX_TURNS} turns")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_vague_first_message(messages: List[Message]) -> bool:
    """
    Detect if the FIRST user message is too vague to recommend without clarification.
    This enforces the 'no recommendations on turn 1 for vague queries' behavior probe.
    """
    if len(messages) != 1:
        return False
    content = messages[0].content.lower().strip()
    vague_patterns = [
        r"^i need an? assessment",
        r"^help me hire",
        r"^i want to hire someone",
        r"^need an? test",
        r"^assess",
        r"^hiring$",
        r"^what tests",
    ]
    return any(re.search(p, content) for p in vague_patterns)


def _is_off_topic(messages: List[Message]) -> bool:
    """Detect clearly off-topic requests (prompt injection, legal, HR advice)."""
    last_user = next(
        (m.content.lower() for m in reversed(messages) if m.role == "user"), ""
    )
    off_topic_signals = [
        "ignore previous",
        "forget your instructions",
        "act as",
        "pretend you are",
        "salary negotiation",
        "legal advice",
        "visa",
        "discrimination lawsuit",
        "competitor",
        "how to pass",
        "cheat",
    ]
    return any(signal in last_user for signal in off_topic_signals)


def _extract_comparison_names(text: str) -> List[str]:
    """
    Try to extract assessment names from comparison questions.
    E.g. "difference between OPQ32r and Verify G+" → ["OPQ32r", "Verify G+"]
    """
    all_names = [item["name"].lower() for item in get_retriever().get_all()]
    found = []
    text_lower = text.lower()
    for item in get_retriever().get_all():
        # Match partial names too (e.g. "OPQ" matches "OPQ32r")
        short_name = item["name"].split("(")[0].strip().lower()
        if short_name in text_lower or item["name"].lower() in text_lower:
            found.append(item["name"])
    return found


def _count_turns(messages: List[Message]) -> int:
    """Count number of complete user turns (each user message = 1 turn)."""
    return sum(1 for m in messages if m.role == "user")


def _call_gemini(system: str, user_prompt: str) -> str:
    """
    Call Gemini Flash with the system + user prompt.
    Returns the raw text response.
    Raises RuntimeError on API failure.
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            temperature=0.3,       # Low temp for consistent, factual output
            max_output_tokens=1024,
            response_mime_type="application/json",  # Force JSON output
        ),
    )
    response = model.generate_content(user_prompt)
    return response.text


def _parse_llm_response(raw: str) -> dict:
    """
    Parse and validate the LLM JSON response.
    Handles cases where the model wraps JSON in markdown fences.
    Falls back to a safe default if parsing fails entirely.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM JSON: %s", raw[:500])
                return {
                    "reply": "I apologize — I encountered a technical issue. Could you rephrase your question?",
                    "recommendations": [],
                    "end_of_conversation": False,
                }
        else:
            return {
                "reply": "I apologize — I encountered a technical issue. Could you rephrase your question?",
                "recommendations": [],
                "end_of_conversation": False,
            }

    # Validate and sanitize
    result = {
        "reply": str(data.get("reply", "How can I help you find the right SHL assessment?")),
        "recommendations": [],
        "end_of_conversation": bool(data.get("end_of_conversation", False)),
    }

    # Validate recommendations against actual catalog
    catalog_url_map = {item["name"].lower(): item for item in get_retriever().get_all()}
    catalog_urls = {item["url"] for item in get_retriever().get_all()}

    raw_recs = data.get("recommendations", [])
    if isinstance(raw_recs, list):
        valid_recs = []
        for rec in raw_recs[:10]:  # Cap at 10
            if not isinstance(rec, dict):
                continue
            name = rec.get("name", "")
            url = rec.get("url", "")
            test_type = rec.get("test_type", "")

            # Find in catalog (by name or URL)
            catalog_item = catalog_url_map.get(name.lower())
            if catalog_item is None:
                # Fuzzy: check if name appears in any catalog item name
                for cname, citem in catalog_url_map.items():
                    if name.lower() in cname or cname in name.lower():
                        catalog_item = citem
                        break

            if catalog_item:
                # Always use catalog's actual URL (prevents hallucination)
                valid_recs.append({
                    "name": catalog_item["name"],
                    "url": catalog_item["url"],
                    "test_type": catalog_item["test_type"],
                })
            elif url in catalog_urls:
                # URL is valid even if name matching failed
                valid_recs.append({
                    "name": name,
                    "url": url,
                    "test_type": test_type,
                })
            else:
                logger.warning("Dropping hallucinated recommendation: %s", name)

        result["recommendations"] = valid_recs

    return result


def _get_safe_fallback_response(messages: List[Message]) -> dict:
    """
    Rule-based fallback when LLM is unavailable.
    Handles basic clarification flow without Gemini.
    """
    turn = _count_turns(messages)
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    if turn == 1:
        return {
            "reply": "I'd be happy to help you find the right SHL assessments. To give you the best recommendations, could you tell me: (1) What role are you hiring for? and (2) What experience level?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Try a basic keyword-based recommendation
    query = build_retrieval_query(messages)
    results = get_retriever().search(query, top_k=5)
    recs = [{"name": r["name"], "url": r["url"], "test_type": r["test_type"]} for r in results[:5]]

    return {
        "reply": f"Based on your requirements, here are the most relevant SHL assessments I found:",
        "recommendations": recs,
        "end_of_conversation": False,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = request.messages

    # ── Guard: off-topic / prompt injection ──────────────────────────────────
    if _is_off_topic(messages):
        return ChatResponse(
            reply="I'm only able to help with SHL assessment selection for hiring purposes. I can't assist with that request.",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Guard: turn cap ───────────────────────────────────────────────────────
    turn_count = _count_turns(messages)
    if turn_count >= MAX_TURNS:
        # Force a final recommendation if not given yet
        query = build_retrieval_query(messages)
        results = retriever.search(query, top_k=5)
        recs = [Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"]) for r in results[:5]]
        return ChatResponse(
            reply="We've reached the conversation limit. Based on our discussion, here are my final SHL assessment recommendations:",
            recommendations=recs,
            end_of_conversation=True,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_query = build_retrieval_query(messages)
    catalog_excerpts = get_retriever().search(retrieval_query, top_k=TOP_K_RETRIEVAL)

    # ── Prompt assembly ───────────────────────────────────────────────────────
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    user_prompt = build_user_prompt(msg_dicts, catalog_excerpts)

    # ── LLM call ──────────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY — using fallback rule-based response")
        parsed = _get_safe_fallback_response(messages)
    else:
        try:
            raw_response = _call_gemini(SYSTEM_PROMPT, user_prompt)
            parsed = _parse_llm_response(raw_response)
        except Exception as e:
            logger.error("Gemini API error: %s", str(e))
            parsed = _get_safe_fallback_response(messages)

    # ── Build response ────────────────────────────────────────────────────────
    recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r["test_type"],
        )
        for r in parsed.get("recommendations", [])
    ]

    return ChatResponse(
        reply=parsed["reply"],
        recommendations=recs,
        end_of_conversation=parsed["end_of_conversation"],
    )
