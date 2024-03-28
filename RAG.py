from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

from create_vector_store import create_vector_store

@st.cache_resource
def create_rag_chain():
    retriever = create_vector_store()
    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain
