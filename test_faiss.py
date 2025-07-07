# test_faiss.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Step 1: Load the index
index = faiss.read_index("faiss_index.idx")

# ✅ Step 2: Load the same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Step 3: Your documents (same order used when building the index)
with open("flight_docs.txt", "r", encoding="utf-8") as f:
    documents = f.read().split('\n\n')

# ✅ Step 4: Ask user
question = input("❓ Enter your question: ")

# ✅ Step 5: Convert question to embedding
query_embedding = model.encode([question])
query_embedding = np.array(query_embedding)

# ✅ Step 6: Search
k = 1  # number of top matches
distances, indices = index.search(query_embedding, k)

# ✅ Step 7: Show answer
print("✅ Top result:")
print(documents[indices[0][0]])