import os
import psycopg2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

class VectorConfig:
    """Configuration for vector operations"""
    HF_TOKEN = os.getenv("HF_TOKEN")
    PDF_FOLDER = "Pdfs"   # Make relative for deployment
    EMBEDDING_MODEL = "BAAI/bge-small-en"
    CHUNK_SIZE = 512

    # PostgreSQL credentials
    PG_HOST = os.getenv("PG_HOST")
    PG_PORT = os.getenv("PG_PORT")
    PG_USER = os.getenv("PG_USER")
    PG_PASSWORD = os.getenv("PG_PASSWORD")
    PG_DATABASE = os.getenv("PG_DATABASE")

def get_embeddings():
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )


def get_pg_connection():
    """Create a PostgreSQL connection"""
    return psycopg2.connect(
        host=VectorConfig.PG_HOST,
        port=VectorConfig.PG_PORT,
        user=VectorConfig.PG_USER,
        password=VectorConfig.PG_PASSWORD,
        database=VectorConfig.PG_DATABASE
    )


def build_vector_db():
    """
    Build PostgreSQL vector DB from PDFs.
    """
    conn = get_pg_connection()
    cursor = conn.cursor()

    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    print("Processing PDFs and inserting embeddings into PostgreSQL...")

    for pdf_file in os.listdir(VectorConfig.PDF_FOLDER):
        if pdf_file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(VectorConfig.PDF_FOLDER, pdf_file))
            pages = loader.load()

            for page in pages:
                chunks = splitter.split_text(page.page_content)

                for chunk in chunks:
                    vector = embeddings.embed_query(chunk)

                    cursor.execute("""
                        INSERT INTO documents (content, embedding, source)
                        VALUES (%s, %s, %s);
                    """, (chunk, vector, pdf_file))

                    conn.commit()

    cursor.close()
    conn.close()
    print("Vector database build completed!")


def search_similar(query, top_k=5):
    """
    Search similar chunks in PostgreSQL.
    Returns list of { content, source }
    """
    conn = get_pg_connection()
    cursor = conn.cursor()

    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    cursor.execute("""
        SELECT content, source
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (query_vector, top_k))


    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return [{"content": r[0], "source": r[1]} for r in rows]
