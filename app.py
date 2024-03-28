from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from RAG import create_rag_chain





def main():
    load_dotenv()
    st.set_page_config(page_title="Ask New Wave Guidelines 2.0" )
    st.header("Ask New Wave Guidelines 2.0" )
    
    
    
    chain = create_rag_chain()
    # show user input
    user_question = st.text_input("Ask a question about your New Wave Guidelines:")
    if user_question:
      response = chain.invoke(user_question)
          
      st.write(response)





if __name__ == '__main__':
    main()
