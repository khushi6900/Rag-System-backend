import os
import re
import asyncio
import json
from typing import List
from openai import OpenAI
from vector_operations import VectorConfig, search_similar, get_embeddings, build_vector_db

class PostgresVectorStore:
    """
    Wrapper for PostgreSQL-based vector store to mimic FAISS interface
    """
    def similarity_search(self, query: str, k: int = 5):
        return search_similar(query, top_k=k)

    def add_documents(self, file_paths: List[str]):
        """
        Insert new PDF documents into PostgreSQL vector DB
        """
        embeddings = get_embeddings()
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=VectorConfig.CHUNK_SIZE, chunk_overlap=50)

        conn = None
        cursor = None
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=VectorConfig.PG_HOST,
                port=VectorConfig.PG_PORT,
                user=VectorConfig.PG_USER,
                password=VectorConfig.PG_PASSWORD,
                database=VectorConfig.PG_DATABASE
            )
            cursor = conn.cursor()

            for path in file_paths:
                loader = PyPDFLoader(path)
                pages = loader.load()
                for page in pages:
                    chunks = splitter.split_text(page.page_content)
                    for chunk in chunks:
                        vector = embeddings.embed_query(chunk)
                        cursor.execute("""
                            INSERT INTO documents (content, embedding, source)
                            VALUES (%s, %s, %s);
                        """, (chunk, vector, os.path.basename(path)))
            conn.commit()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

class RAGSystem:
    """
    RAG Pipeline with PostgreSQL vector store and streaming support
    """

    def __init__(self, recreate_index: bool = False):
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=VectorConfig.HF_TOKEN
        )

        # Initialize vector store
        self.vector_store = PostgresVectorStore()

        # Optionally rebuild vector DB from PDFs
        if recreate_index:
            build_vector_db()

    async def stream_query(self, question: str, k: int = 3, stream_delay: float = 0.5):
        """
        Execute RAG query with streaming response
        """
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=k)

            if not docs:
                # No relevant documents found
                yield f"data: {json.dumps({'token': '', 'sources': [], 'done': True, 'full_response': 'No relevant documents found in the database.'})}\n\n"
                return

            context = "\n\n".join([doc["content"] for doc in docs])
            sources = list(set([doc["source"] for doc in docs]))

            # Generate answer using LLM with streaming
            completion = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are MediRAG, a specialized Medical Assistant designed to analyze and summarize medical literature. 

IMPORTANT IDENTITY GUIDELINES:
1. You are NOT ChatGPT - you are MediRAG, a specialized medical research tool
2. Never introduce yourself as ChatGPT or any OpenAI product
3. When asked about your identity, state: "I am MediRAG, your Medical Assistant"

CONTEXT-BASED RESPONSE RULES:
- Answer using ONLY the provided context: {context}
- If the context doesn't contain relevant information, clearly state this
- Critically appraise medical literature and summarize findings for clinical practice
- Provide answers in plain text format without markdown
- Use clear, readable formatting with paragraphs and proper punctuation
- Never hallucinate or invent information not present in the context"""
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                stream=True
            )

            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    clean_token = self._clean_markdown(token)
                    yield f"data: {json.dumps({'token': clean_token, 'sources': sources, 'done': False})}\n\n"
                    await asyncio.sleep(stream_delay)

            # Final message
            yield f"data: {json.dumps({'token': '', 'sources': sources, 'done': True, 'full_response': full_response})}\n\n"

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting"""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\|.*?\|\n\|.*?\|', '', text)
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
        return text

    def add_documents(self, file_paths: List[str]):
        """Add new PDFs to the PostgreSQL vector store"""
        self.vector_store.add_documents(file_paths)
