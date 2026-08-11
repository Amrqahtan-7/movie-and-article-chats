import requests
from typing import List


class HFHostedEmbeddings:
    """
    Drop-in replacement for the local HuggingFaceEmbeddings class.
    Calls Hugging Face's hosted inference API instead of loading the model
    (and PyTorch) into local memory — this is what makes it fit in Render's
    free-tier 512MB RAM limit.

    Uses the SAME model as before (all-MiniLM-L6-v2), so vectors stay
    dimensionally compatible with your existing Qdrant collection — no
    need to rebuild it.
    """

    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
        response.raise_for_status()
        result = response.json()

        # This model returns per-token embeddings; average them into one
        # vector per input to match what sentence-transformers does locally
        # (mean pooling).
        embeddings = []
        for item in result:
            if isinstance(item[0], list):  # token-level output, needs pooling
                vector_len = len(item[0])
                pooled = [sum(token[i] for token in item) / len(item) for i in range(vector_len)]
                embeddings.append(pooled)
            else:  # already a single pooled vector
                embeddings.append(item)
        return embeddings

    # ---- Methods LangChain's vectorstore code expects ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]