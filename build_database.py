import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv("data/wiki_movie_plots_deduped.csv")
df = df[["Title", "Plot"]].dropna()
df = df.head(5000)  # adjust or remove once ready for the full dataset

# 3. Convert to LangChain documents
docs = [
    Document(page_content=row["Plot"], metadata={"title": row["Title"]})
    for _, row in df.iterrows()
]

# 4. Reset Qdrant collection
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
try:
    client.delete_collection("movies_demo")
except Exception:
    pass

# 5. Embed + upload to Qdrant
print(f"Embedding {len(docs)} documents and uploading to Qdrant...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="movies_demo",
    batch_size=64,
    timeout=120,
)

print("Done! Your Qdrant collection 'movies_demo' is ready.")
print("You can now run chat.py anytime without repeating this step.")