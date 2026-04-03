import faiss
import numpy as np


class Retriever:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks

        passages = ["passage: " + c for c in chunks]

        emb = self.embedding_model.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        np.save("data/processed/embeddings.npy", emb)

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def save(self):
        faiss.write_index(self.index, "data/processed/index.faiss")
        np.save("data/processed/chunks.npy", self.chunks)

    def load(self):
        self.index = faiss.read_index("data/processed/index.faiss")
        self.chunks = np.load("data/processed/chunks.npy", allow_pickle=True)

    def search(self, query, k=6):
        q = self.embedding_model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, idx = self.index.search(q, k)

        context = []
        sources = []

        for i, s in zip(idx[0], scores[0]):
            if s > 0.2:
                context.append(self.chunks[i])
                sources.append(f"Chunk {i} ({s:.2f})")

        if not context:
            context = [self.chunks[i] for i in idx[0][:3]]

        return "\n".join(context), sources