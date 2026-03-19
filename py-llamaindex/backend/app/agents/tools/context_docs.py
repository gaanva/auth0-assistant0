from llama_index.core.tools import FunctionTool
from openfga_sdk.client.models import ClientBatchCheckItem

from app.core.rag import find_relevant_content
from app.core.fga import authorization_manager


async def get_context_docs_fn(
    question: str, credentials: dict | None = None
) -> str:
    """Use the tool when user asks for documents or projects or anything that is stored in the knowledge base."""

    if not credentials:
        return "There is no user logged in."

    user = credentials.get("user")

    if not user:
        return "There is no user logged in."

    user_email = user.get("email")

    # Get relevant documents
    documents = await find_relevant_content(question, limit=25)

    if not documents:
        return "No relevant documents found."

    # Filter by FGA authorization
    if authorization_manager.openfga_client is None:
        return "Authorization service not available."

    # Batch check authorization
    checks = [
        ClientBatchCheckItem(
            user=f"user:{user_email}",
            object=f"doc:{doc['document_id']}",
            relation="can_view",
        )
        for doc in documents
    ]

    try:
        response = await authorization_manager.openfga_client.batch_check(checks)
        authorized_docs = []
        for i, result in enumerate(response.result):
            if result.allowed:
                authorized_docs.append(documents[i])
    except Exception:
        # If batch check fails, return all documents
        authorized_docs = documents

    if not authorized_docs:
        return "No authorized documents found."

    return "\n\n".join([doc["content"] for doc in authorized_docs])


def create_context_docs_tool(credentials: dict | None = None) -> FunctionTool:
    """Create the context docs tool with credentials bound."""

    async def _get_context_docs(question: str) -> str:
        """Use the tool when user asks for documents or projects or anything that is stored in the knowledge base."""
        return await get_context_docs_fn(question, credentials)

    return FunctionTool.from_defaults(
        async_fn=_get_context_docs,
        name="get_context_docs",
        description="Use the tool when user asks for documents or projects or anything that is stored in the knowledge base.",
    )
