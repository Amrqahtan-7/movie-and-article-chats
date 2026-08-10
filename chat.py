import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# 2. Connect to the EXISTING Qdrant collection (no re-embedding, no re-uploading)
print("Connecting to existing knowledge base...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="movies_demo",
)

# 3. Set up retrieval + chat model
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

# 4. Chat loop
print("\nChatbot ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.invoke({"input": query})
    print("Bot:", response["answer"], "\n")