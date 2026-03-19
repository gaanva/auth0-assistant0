import uuid
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlmodel import Session, select, text
from sqlalchemy import func

from app.core.config import settings
from app.core.db import engine
from app.models.embeddings import Embedding

embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY,
)


def generate_embeddings(
    document_id: uuid.UUID, file_name: str, text_content: str
) -> list[Embedding]:
    """Generate embeddings for a document using LlamaIndex."""
    # Split text into chunks
    chunks = _split_text(text_content, chunk_size=100, chunk_overlap=10)

    embeddings_list = []
    for chunk in chunks:
        embedding = embedding_model.get_text_embedding(chunk)
        embeddings_list.append(
            Embedding(
                document_id=document_id,
                meta={
                    "file_name": file_name,
                    "document_id": str(document_id),
                },
                content=chunk,
                embedding=embedding,
            )
        )

    return embeddings_list


def _split_text(text_content: str, chunk_size: int = 100, chunk_overlap: int = 10) -> list[str]:
    """Split text into chunks with overlap using sentence-aware splitting."""
    if not text_content.strip():
        return []

    sentences = []
    for line in text_content.split("\n"):
        line = line.strip()
        if line:
            sentences.append(line)

    if not sentences:
        return []

    chunks = []
    current_chunk_words: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        current_chunk_words.extend(words)

        while len(current_chunk_words) >= chunk_size:
            chunk_text = " ".join(current_chunk_words[:chunk_size])
            chunks.append(chunk_text)
            current_chunk_words = current_chunk_words[chunk_size - chunk_overlap:]

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


async def find_relevant_content(query: str, limit: int = 4) -> list[dict]:
    """Find relevant content using cosine similarity search."""
    query_embedding = embedding_model.get_text_embedding(query)

    with Session(engine) as db_session:
        # Use pgvector cosine distance operator
        results = db_session.exec(
            text(
                """
                SELECT content, document_id, 1 - (embedding <=> :query_embedding::vector) as similarity
                FROM embedding
                WHERE 1 - (embedding <=> :query_embedding::vector) > 0.5
                ORDER BY similarity DESC
                LIMIT :limit
                """
            ).bindparams(
                query_embedding=str(query_embedding),
                limit=limit,
            )
        ).all()

        return [
            {
                "content": row[0],
                "document_id": str(row[1]),
                "similarity": float(row[2]),
            }
            for row in results
        ]
