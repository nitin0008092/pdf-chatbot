from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Load PDF
loader = PyPDFLoader("data/StudentLeavePolicy.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Create vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# Save database
vectorstore.save_local("faiss_index")

print("Vector database created successfully!")
