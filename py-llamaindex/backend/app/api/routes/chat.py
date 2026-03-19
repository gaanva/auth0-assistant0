import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.auth import auth_client
from app.agents.assistant0 import create_agent

agent_router = APIRouter(prefix="/agent", tags=["agent"])

# In-memory thread storage
_threads: dict[str, dict] = {}


def _get_credentials(auth_session: dict) -> dict:
    return {
        "access_token": auth_session.get("token_sets", [{}])[0].get("access_token"),
        "refresh_token": auth_session.get("refresh_token"),
        "user": auth_session.get("user"),
    }


def _ensure_thread(thread_id: str) -> dict:
    if thread_id not in _threads:
        _threads[thread_id] = {
            "thread_id": thread_id,
            "created_at": "",
            "updated_at": "",
            "metadata": {},
            "status": "idle",
            "values": {"messages": []},
        }
    return _threads[thread_id]


@agent_router.post("/threads")
async def create_thread(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    thread_id = str(uuid.uuid4())
    thread = _ensure_thread(thread_id)
    return JSONResponse(content=thread)


@agent_router.post("/threads/{thread_id}/runs/stream")
async def run_stream(
    thread_id: str,
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    body = await request.json()
    credentials = _get_credentials(auth_session)
    thread = _ensure_thread(thread_id)
    messages = thread["values"].get("messages", [])

    # Extract input messages
    input_data = body.get("input", {})
    new_messages = input_data.get("messages", [])

    # Add new messages to thread
    for msg in new_messages:
        msg_id = str(uuid.uuid4())
        messages.append({
            "type": msg.get("type", "human"),
            "content": msg.get("content", ""),
            "id": msg_id,
        })

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
        yield f"event: metadata\ndata: {json.dumps({'run_id': run_id})}\n\n"

        try:
            handler = agent.run(user_message)
            response = await handler

            # Extract the response content
            response_text = str(response)

            # Create AI message
            ai_msg_id = str(uuid.uuid4())
            ai_message = {
                "type": "ai",
                "content": response_text,
                "id": ai_msg_id,
                "tool_calls": [],
            }
            messages.append(ai_message)

            # Send values event with all messages
            values_data = {
                "messages": messages,
            }
            yield f"event: values\ndata: {json.dumps(values_data)}\n\n"

        except Exception as e:
            error_str = str(e)
            # Check if this is a Token Vault interrupt
            if "token_vault" in error_str.lower() or "interrupt" in error_str.lower():
                # Try to parse interrupt data from the error
                try:
                    interrupt_data = _parse_interrupt(e)
                    if interrupt_data:
                        values_data = {"messages": messages}
                        yield f"event: values\ndata: {json.dumps(values_data)}\n\n"

                        interrupts_payload = [{
                            "value": interrupt_data,
                            "resumable": True,
                            "ns": [f"interrupt_{uuid.uuid4().hex[:8]}"],
                            "when": "during",
                        }]
                        yield f"event: updates\ndata: {json.dumps({'__interrupt': interrupts_payload})}\n\n"

                        thread["values"]["messages"] = messages
                        return
                except Exception:
                    pass

            # Regular error
            error_msg_id = str(uuid.uuid4())
            messages.append({
                "type": "ai",
                "content": f"An error occurred: {error_str}",
                "id": error_msg_id,
            })
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
    thread = _ensure_thread(thread_id)
    return JSONResponse(content={
        "values": thread["values"],
        "next": [],
        "tasks": [],
        "metadata": {},
        "created_at": thread.get("created_at", ""),
        "parent_config": None,
    })


@agent_router.post("/threads/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    thread = _ensure_thread(thread_id)
    return JSONResponse(content={"configurable": {"thread_id": thread_id}})


@agent_router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    auth_session=Depends(auth_client.require_session),
):
    thread = _ensure_thread(thread_id)
    return JSONResponse(content=thread)


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
    return JSONResponse(content=[{
        "assistant_id": "agent",
        "graph_id": "agent",
        "name": "Agent",
        "config": {},
        "metadata": {},
    }])


@agent_router.post("/assistants/search")
async def search_assistants_post(
    request: Request,
    auth_session=Depends(auth_client.require_session),
):
    return JSONResponse(content=[{
        "assistant_id": "agent",
        "graph_id": "agent",
        "name": "Agent",
        "config": {},
        "metadata": {},
    }])


def _parse_interrupt(error: Exception) -> dict | None:
    """Try to extract Token Vault interrupt data from an exception."""
    error_str = str(error)
    # Auth0 AI interrupts typically contain connection/scope info
    try:
        # Try to find JSON in the error message
        import re
        json_match = re.search(r'\{.*\}', error_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "connection" in data:
                return data
    except (json.JSONDecodeError, AttributeError):
        pass

    # Check for auth0-ai interrupt attributes
    if hasattr(error, 'interrupt_value'):
        return error.interrupt_value
    if hasattr(error, 'value'):
        return error.value

    return None
