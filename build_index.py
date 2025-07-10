from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load docs
with open("flight_docs.txt", encoding="utf-8") as f:
    content = f.read()

docs = [Document(page_content=chunk) for chunk in content.split('\n\n')]

# Create vector DB
db = FAISS.from_documents(docs, embeddings)

# Save
db.save_local("faiss_index")
print("✅ FAISS index created in faiss_index/")