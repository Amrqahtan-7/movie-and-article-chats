import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# 1. Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv("data/wiki_movie_plots_deduped.csv")
df = df[["Title", "Plot"]].dropna()
df = df.head(5000)  # adjust or remove this line once you're ready for the full dataset

# 3. Convert to LangChain documents
docs = [
    Document(page_content=row["Plot"], metadata={"title": row["Title"]})
    for _, row in df.iterrows()
]

# 4. Reset Qdrant collection (only needed if re-embedding fresh data each run)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
try:
    client.delete_collection("movies_demo")
except Exception:
    pass  # collection may not exist yet, that's fine

# 5. Embed + store in Qdrant (batched, with longer timeout to avoid ReadTimeout errors)
print("Embedding and uploading to Qdrant...")
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

# 6. Set up retrieval + chat model
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_key=HF_API_KEY,
    openai_api_base="https://router.huggingface.co/v1",
)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Chat loop
print("\nChatbot ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.invoke({"input": query})
    print("Bot:", response["answer"], "\n")