# create_faiss_index.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ✅ Step: Load your document
with open("flight_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ✅ Step: Split into chunks
documents = text.split('\n\n')  # split by blank lines
print(f"Loaded {len(documents)} documents.")

# ✅ Step 4.2: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
embeddings = np.array(embeddings)

# ✅ Step 4.3: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ✅ Save index
faiss.write_index(index, "faiss_index.idx")
print("✅ FAISS index created and saved as faiss_index.idx")