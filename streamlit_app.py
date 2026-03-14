import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

st.title("📄 PDF Chatbot")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

query = st.text_input("Ask a question about the document")

if query:
    result = qa.run(query)
    st.write("### Answer")
    st.write(result)
