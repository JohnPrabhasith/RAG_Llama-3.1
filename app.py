import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time


from dotenv import load_dotenv
load_dotenv()



groq_api_key = os.getenv('groq_API')


def vector_embeddings():
    if "vectors" not in st.session_state:
        model_name = 'BAAI/bge-small-en'
        # model_kwags = {"device":"cpu"}
        # encode_kwags = {"normalize_embeddings" : True}
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name = model_name
        )
        st.session_state.loader = PyPDFDirectoryLoader(
            './documents'
        )
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 200,   
        )
        st.session_state.final_document = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_document,
            st.session_state.embeddings
        )



llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = 'Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    '''
    Answer the Questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context> {context} <context>
    Questions : {input}
    '''
)



st.set_page_config(page_title="Chat with your Documents (PDF/TXT)",page_icon='✨')
st.title("ChatBot with Llama3")

prompt1 = st.text_input("Enter the Query from the documents")

if st.button("Run Query!"):
    vector_embeddings()
    st.write("Vector_Store DB is Ready")



if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(
    llm = llm,
    prompt=prompt
    )
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({'input' : prompt1})
    st.write(response['answer'])

    #With StreamLit Expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------------')




