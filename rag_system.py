import os
import re
import json
import asyncio
from typing import List
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from vector_operations import (
    VectorConfig,
    search_similar,
    get_embeddings,
    build_vector_db
)

# ---------------- VECTOR STORE ---------------- #

class PostgresVectorStore:
    def similarity_search(self, query: str, k: int = 5):
        return search_similar(query, top_k=k)

    def add_documents(self, file_paths: List[str]):
        embeddings = get_embeddings()

        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import psycopg2

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=VectorConfig.CHUNK_SIZE,
            chunk_overlap=50
        )

        conn = psycopg2.connect(
            host=VectorConfig.PG_HOST,
            port=VectorConfig.PG_PORT,
            user=VectorConfig.PG_USER,
            password=VectorConfig.PG_PASSWORD,
            database=VectorConfig.PG_DATABASE
        )
        cursor = conn.cursor()

        try:
            for path in file_paths:
                loader = PyPDFLoader(path)
                pages = loader.load()

                for page in pages:
                    chunks = splitter.split_text(page.page_content)
                    for chunk in chunks:
                        vector = embeddings.embed_query(chunk)
                        cursor.execute(
                            """
                            INSERT INTO documents (content, embedding, source)
                            VALUES (%s, %s, %s);
                            """,
                            (chunk, vector, os.path.basename(path))
                        )
            conn.commit()
        finally:
            cursor.close()
            conn.close()


# ---------------- RAG SYSTEM ---------------- #

class RAGSystem:
    def __init__(self, recreate_index: bool = False):
        # HuggingFace LLM
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("HF_MODEL_ID"),
            task="text-generation",
            huggingfacehub_api_token=VectorConfig.HF_TOKEN,
            max_new_tokens=512,
            temperature=0.7
        )

        self.chat_model = ChatHuggingFace(llm=llm)
        self.vector_store = PostgresVectorStore()

        if recreate_index:
            build_vector_db()

    async def stream_query(self, question: str, k: int = 3, stream_delay: float = 0.05):
        try:
            yield f"data: {json.dumps({'status': 'stream_started'})}\n\n"

            docs = self.vector_store.similarity_search(question, k=k)

            if not docs:
                yield f"data: {json.dumps({'done': True, 'full_response': 'No relevant documents found.', 'sources': []})}\n\n"
                return

            context = "\n\n".join(doc["content"] for doc in docs)
            sources = list(set(doc["source"] for doc in docs))

            prompt = f"""
You are MediRAG, a specialized Medical Assistant.

Rules:
- Use ONLY the context below
- If answer is not in context, say so clearly
- Do NOT mention ChatGPT or OpenAI
- Plain text only

Context:
{context}

Question:
{question}
"""

            full_response = ""

            async for chunk in self.chat_model.astream(prompt):
                token = chunk.content
                if token:
                    clean = self._clean_markdown(token)
                    full_response += clean

                    yield f"data: {json.dumps({'token': clean, 'done': False, 'sources': sources})}\n\n"
                    await asyncio.sleep(stream_delay)

            yield f"data: {json.dumps({'done': True, 'full_response': full_response, 'sources': sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    def add_documents(self, file_paths: List[str]):
        self.vector_store.add_documents(file_paths)

    def _clean_markdown(self, text: str) -> str:
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
        return text