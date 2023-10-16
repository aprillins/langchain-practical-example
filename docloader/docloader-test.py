from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import chromadb

persisten_client = chromadb.PersistentClient("../chroma_langchain_db")

vectorstore = Chroma(
    client=persisten_client,
    collection_name="trading_data",
    persist_directory="../chroma_langchain_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)

llm = ChatOpenAI(model="gpt-4o-mini")

retriever = vectorstore.as_retriever()
# result = retriever.invoke("get me the latest 3 price data of bmri stock price. And give me your suggestion")

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke('get me the latest data of jpfa on 10/25/2024'))