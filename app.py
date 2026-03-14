from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# Ask questions
while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break

    result = qa.run(query)
    print("\nAnswer:", result)
