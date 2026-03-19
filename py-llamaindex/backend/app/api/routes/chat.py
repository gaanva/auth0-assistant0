import json
import logging
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult, AgentOutput
from llama_index.core.memory import Memory

logger = logging.getLogger(__name__)

from auth0_ai.interrupts.auth0_interrupt import Auth0Interrupt
from auth0_ai_llamaindex.context import set_ai_context

from app.core.auth import auth_client
from app.core.auth0_ai import current_credentials
from app.agents.assistant0 import create_agent

agent_router = APIRouter(prefix="/agent", tags=["agent"])

# In-memory thread storage, scoped by user sub
_threads: dict[str, dict[str, dict]] = {}  # user_sub -> thread_id -> thread
_memories: dict[str, dict[str, Memory]] = {}  # user_sub -> thread_id -> Memory


def _get_credentials(auth_session: dict) -> dict:
    return {
        "access_token": auth_session.get("token_sets", [{}])[0].get("access_token"),
        "refresh_token": auth_session.get("refresh_token"),
        "user": auth_session.get("user"),
    }


def _get_user_sub(auth_session: dict) -> str:
    return auth_session.get("user", {}).get("sub", "anonymous")


def _ensure_thread(user_sub: str, thread_id: str) -> dict:
    if user_sub not in _threads:
        _threads[user_sub] = {}
    if thread_id not in _threads[user_sub]:
        _threads[user_sub][thread_id] = {
            "thread_id": thread_id,
            "created_at": "",
            "updated_at": "",
            "metadata": {},
            "status": "idle",
            "values": {"messages": []},
        }
    return _threads[user_sub][thread_id]


def _ensure_memory(user_sub: str, thread_id: str) -> Memory:
    if user_sub not in _memories:
        _memories[user_sub] = {}
    if thread_id not in _memories[user_sub]:
        _memories[user_sub][thread_id] = Memory.from_defaults(
            session_id=thread_id,
        )
    return _memories[user_sub][thread_id]


@agent_router.post("/threads")
async def create_thread(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    user_sub = _get_user_sub(auth_session)
    thread_id = str(uuid.uuid4())
    thread = _ensure_thread(user_sub, thread_id)
    return JSONResponse(content=thread)


@agent_router.post("/threads/{thread_id}/runs/stream")
async def run_stream(
    thread_id: str,
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    body = await request.json()
    credentials = _get_credentials(auth_session)
    user_sub = _get_user_sub(auth_session)
    thread = _ensure_thread(user_sub, thread_id)
    messages = thread["values"].get("messages", [])
    memory = _ensure_memory(user_sub, thread_id)

    # Set credentials for auth0-ai Token Vault and CIBA flows
    current_credentials.set(credentials)

    # Set auth0-ai-llamaindex thread context
    set_ai_context(thread_id)

    # Extract input messages
    input_data = body.get("input", {})
    new_messages = input_data.get("messages", [])

    # Add new messages to thread
    for msg in new_messages:
        msg_id = str(uuid.uuid4())
        messages.append(
            {
                "type": msg.get("type", "human"),
                "content": msg.get("content", ""),
                "id": msg_id,
            }
        )

    # Get the last human message for the agent
    user_message = ""
    for msg in reversed(messages):
        if msg.get("type") == "human":
            user_message = msg.get("content", "")
            break

    agent = create_agent(credentials)

    async def event_stream():
        # Send metadata event
        run_id = str(uuid.uuid4())
        ai_msg_id = str(uuid.uuid4())
        yield f"event: metadata\ndata: {json.dumps({'run_id': run_id})}\n\n"

        try:
            handler = agent.run(user_message, memory=memory)
            tool_calls_seen: list[dict] = []

            # Stream events to capture tool calls and token deltas
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    if event.delta:
                        partial_msg = {
                            "type": "ai",
                            "content": event.response,
                            "id": ai_msg_id,
                            "tool_calls": [],
                        }
                        metadata = {"run_id": run_id}
                        yield f"event: messages/partial\ndata: {json.dumps([partial_msg, metadata])}\n\n"
                elif isinstance(event, ToolCallResult):
                    if event.tool_output.is_error:
                        logger.error(
                            "Tool '%s' failed: %s",
                            event.tool_name,
                            event.tool_output.content,
                        )
                elif isinstance(event, ToolCall):
                    tool_calls_seen.append(
                        {
                            "name": event.tool_name,
                            "args": event.tool_kwargs,
                            "id": getattr(event, "tool_id", str(uuid.uuid4())),
                            "type": "tool_call",
                        }
                    )

            response = await handler
            response_text = str(response)

            # Send complete message event
            ai_message = {
                "type": "ai",
                "content": response_text,
                "id": ai_msg_id,
                "tool_calls": tool_calls_seen,
            }
            metadata = {"run_id": run_id}
            yield f"event: messages/complete\ndata: {json.dumps([ai_message, metadata])}\n\n"

            messages.append(ai_message)

            # Send values event with all messages
            values_data = {"messages": messages}
            yield f"event: values\ndata: {json.dumps(values_data)}\n\n"

        except Auth0Interrupt as e:
            # Structured interrupt from auth0-ai (Token Vault, async authorization)
            interrupt_data = e.to_json()
            values_data = {"messages": messages}
            yield f"event: values\ndata: {json.dumps(values_data)}\n\n"

            interrupts_payload = [
                {
                    "value": interrupt_data,
                    "resumable": True,
                    "ns": [f"interrupt_{uuid.uuid4().hex[:8]}"],
                    "when": "during",
                }
            ]
            yield f"event: updates\ndata: {json.dumps({'__interrupt': interrupts_payload})}\n\n"

            thread["values"]["messages"] = messages
            return

        except Exception as e:
            error_msg_id = str(uuid.uuid4())
            messages.append(
                {
                    "type": "ai",
                    "content": f"An error occurred: {e}",
                    "id": error_msg_id,
                }
            )
            values_data = {"messages": messages}
            yield f"event: values\ndata: {json.dumps(values_data)}\n\n"

        thread["values"]["messages"] = messages
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@agent_router.get("/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    auth_session=Depends(auth_client.require_session),
):
    user_sub = _get_user_sub(auth_session)
    thread = _ensure_thread(user_sub, thread_id)
    return JSONResponse(
        content={
            "values": thread["values"],
            "next": [],
            "tasks": [],
            "metadata": {},
            "created_at": thread.get("created_at", ""),
            "parent_config": None,
        }
    )


@agent_router.post("/threads/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    user_sub = _get_user_sub(auth_session)
    _ensure_thread(user_sub, thread_id)
    return JSONResponse(content={"configurable": {"thread_id": thread_id}})


@agent_router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    auth_session=Depends(auth_client.require_session),
):
    user_sub = _get_user_sub(auth_session)
    thread = _ensure_thread(user_sub, thread_id)
    return JSONResponse(content=thread)


@agent_router.post("/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    user_sub = _get_user_sub(auth_session)
    thread = _ensure_thread(user_sub, thread_id)
    # Return current state as a single history entry
    return JSONResponse(
        content=[
            {
                "values": thread["values"],
                "next": [],
                "tasks": [],
                "metadata": {},
                "created_at": thread.get("created_at", ""),
                "parent_config": None,
                "checkpoint": {"thread_id": thread_id},
            }
        ]
    )


@agent_router.post("/threads/search")
async def search_threads(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    return JSONResponse(content=[])


@agent_router.get("/assistants/search")
async def search_assistants(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    return JSONResponse(
        content=[
            {
                "assistant_id": "agent",
                "graph_id": "agent",
                "name": "Agent",
                "config": {},
                "metadata": {},
            }
        ]
    )


@agent_router.post("/assistants/search")
async def search_assistants_post(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    return JSONResponse(
        content=[
            {
                "assistant_id": "agent",
                "graph_id": "agent",
                "name": "Agent",
                "config": {},
                "metadata": {},
            }
        ]
    )
