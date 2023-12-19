import os
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import time
from langchain import OpenAI

from dotenv import load_dotenv
load_dotenv() 

st.title("FINALYTICS, the AI-driven Tool for Financial Insights")
st.sidebar.title("Financial Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Click to Process")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
  
    # load data
  
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Activated...")
    data = loader.load()
  
    # split data
  
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Activated...")
    docs = text_splitter.split_documents(data)
  
    # create embeddings and save it to FAISS index
  
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vector...")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Ask anything: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
          
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
          
            st.header("Answer for the query")
            st.write(result["answer"])

            # Display sources, if available
          
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
