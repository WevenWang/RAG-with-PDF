path = "assets/new_wave_guideline.pdf"

from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

class Element(BaseModel):
    type: str
    text: Any

# Get elements
def get_raw_pdf_elements(path: str) -> Any:
    raw_pdf_elements = partition_pdf(
        filename=path,
        # Unstructured first finds embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )
    return raw_pdf_elements

# Categorize elements
def categorize_elements(raw_pdf_elements: Any) -> Any:
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
    return categorized_elements

categorized_elements = categorize_elements(get_raw_pdf_elements(path))

table_elements = [e for e in categorized_elements if e.type == "table"]
print('# of Tables:', len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print('# of Text:', len(text_elements))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo" , openai_api_key="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")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

# Apply to texts
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})



import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def create_vector_store():
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings(openai_api_key="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"))

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    return retriever
