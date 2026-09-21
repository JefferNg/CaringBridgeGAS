"""
Local proxy for the CaringBridge writing assistant.

Uses Application Default Credentials (ADC) — no API keys in the browser.
Run once:  gcloud auth application-default login
Then:      uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Optional

import vertexai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "341734686561")
LOCATION = os.environ.get("GCP_LOCATION", "us-west1")
AGENT_ID = os.environ.get("GCP_AGENT_ID", "8909531835868905472")

RESOURCE_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_ID}"
)

app = FastAPI(title="CaringBridge GAS agent proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: Optional[Any] = None
_agent: Optional[Any] = None

# Bound onto the remote AgentEngine when Vertex omitted class_methods metadata.
_ADK_FALLBACK_SCHEMAS = [
    {"name": "async_create_session", "api_mode": "async"},
    {"name": "async_stream_query", "api_mode": "async_stream"},
    {"name": "stream_query", "api_mode": "stream"},
]


def _agent_is_deployed(agent: Any) -> bool:
    """True when the Reasoning Engine has something that can actually run."""
    spec = getattr(getattr(agent, "api_resource", None), "spec", None)
    if spec is None:
        return False
    for field in ("package_spec", "source_code_spec", "container_spec"):
        if getattr(spec, field, None) is not None:
            return True
    return False


def _ensure_query_methods(agent: Any) -> None:
    """Attach stream/query helpers when get() returned a metadata-only object.

    Vertex only binds methods from spec.class_methods. Agent Studio / Terraform
    deploys often leave that field empty, so async_stream_query is missing even
    though the container still serves it.
    """
    if hasattr(agent, "async_stream_query") or hasattr(agent, "stream_query"):
        return
    try:
        from vertexai._genai import _agent_engines_utils as utils
    except Exception:
        return

    schemas = None
    if hasattr(agent, "operation_schemas"):
        try:
            schemas = agent.operation_schemas()
        except Exception:
            schemas = None
    if not schemas:
        schemas = _ADK_FALLBACK_SCHEMAS

    # operation_schemas() may be empty; temporarily fake it for registration.
    original = getattr(agent, "operation_schemas", None)

    def _schemas():
        return schemas

    try:
        agent.operation_schemas = _schemas  # type: ignore[method-assign]
        utils._register_api_methods_or_raise(agent_engine=agent)
    except Exception:
        pass
    finally:
        if original is not None:
            agent.operation_schemas = original  # type: ignore[method-assign]


def get_agent():
    global _client, _agent
    if _agent is not None:
        return _agent
    _client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    _agent = _client.agent_engines.get(name=RESOURCE_NAME)
    if not _agent_is_deployed(_agent):
        raise RuntimeError(
            f"Agent Engine {RESOURCE_NAME} has no package_spec/source_code_spec/"
            "container_spec. It looks like an Agent Studio identity shell, not a "
            "deployed runnable agent. Deploy/publish the agent (ADK CLI, "
            "agent_engines.create, or Agent Engine with a package), then retry."
        )
    _ensure_query_methods(_agent)
    if not (
        hasattr(_agent, "async_stream_query") or hasattr(_agent, "stream_query")
    ):
        raise RuntimeError(
            "Agent Engine loaded but has no stream_query/async_stream_query. "
            "Redeploy with class_methods that include async_stream_query "
            '(api_mode="async_stream").'
        )
    return _agent


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    draft: str = ""
    user_id: str = "caringbridge_user"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    action: Optional[str] = None
    text: Optional[str] = None


def build_prompt(user_message: str, draft: str) -> str:
    draft_block = draft.strip() or "(empty)"
    return "\n".join(
        [
            "You are helping someone write a CaringBridge health update post.",
            "You can either ask a clarifying question, or update the post draft.",
            "",
            "Current post draft:",
            '"""',
            draft_block,
            '"""',
            "",
            "User message:",
            user_message,
            "",
            "Reply with ONLY a single JSON object (no markdown fences):",
            'If you need more info: {"action":"ask","message":"<question or reply>"}',
            'If you are ready to edit the post: {"action":"edit","text":"<full updated post>","message":"<brief note to the user>"}',
        ]
    )


def collect_text(events: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for event in events:
        parts = []
        if isinstance(event, dict):
            content = event.get("content") or {}
            if isinstance(content, dict):
                parts = content.get("parts") or []
            parts = parts or event.get("parts") or []
            if isinstance(event.get("text"), str):
                chunks.append(event["text"])
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks).strip()


def parse_structured(reply: str) -> tuple[str, Optional[str], Optional[str]]:
    """Return (message, action, edited_text)."""
    import json
    import re

    candidates: list[str] = [reply.strip()]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", reply, re.I)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = reply.find("{"), reply.rfind("}")
    if start != -1 and end > start:
        candidates.append(reply[start : end + 1])

    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict) or "action" not in data:
            continue
        action = data.get("action")
        if action == "edit" and isinstance(data.get("text"), str):
            return (
                str(data.get("message") or "I updated your post in the editor."),
                "edit",
                data["text"],
            )
        if action == "ask" and data.get("message"):
            return str(data["message"]), "ask", None

    return reply, None, None


@app.get("/api/health")
def health():
    info: dict[str, Any] = {
        "ok": True,
        "project": PROJECT_ID,
        "location": LOCATION,
        "agent": AGENT_ID,
        "resource": RESOURCE_NAME,
    }
    try:
        agent = get_agent()
        info["deployed"] = True
        info["has_async_stream_query"] = hasattr(agent, "async_stream_query")
        info["has_stream_query"] = hasattr(agent, "stream_query")
    except Exception as exc:
        info["ok"] = False
        info["deployed"] = False
        info["error"] = str(exc)
    return info


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        agent = get_agent()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load the Vertex agent. "
                "Run `gcloud auth application-default login` and confirm "
                f"project/location/agent ({RESOURCE_NAME}). Error: {exc}"
            ),
        ) from exc

    session_id = body.session_id
    try:
        if not session_id and hasattr(agent, "async_create_session"):
            session = await agent.async_create_session(user_id=body.user_id)
            if isinstance(session, dict):
                session_id = session.get("id") or session.get("session_id")
            else:
                session_id = getattr(session, "id", None)

        events: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "user_id": body.user_id,
            "message": build_prompt(body.message, body.draft),
        }
        if session_id:
            kwargs["session_id"] = session_id

        if hasattr(agent, "async_stream_query"):
            async for event in agent.async_stream_query(**kwargs):
                if isinstance(event, dict):
                    events.append(event)
                else:
                    events.append({"text": str(event)})
        else:
            # Sync stream fallback for engines that only expose stream_query.
            for event in agent.stream_query(**kwargs):
                if isinstance(event, dict):
                    events.append(event)
                else:
                    events.append({"text": str(event)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent query failed: {exc}") from exc

    raw = collect_text(events) or "(empty agent reply)"
    message, action, edited = parse_structured(raw)
    return ChatResponse(
        reply=message,
        session_id=session_id,
        action=action,
        text=edited,
    )


STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
