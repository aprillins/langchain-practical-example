from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

loader = DirectoryLoader("../data", glob="**/*.csv", loader_cls=CSVLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma.from_documents(
    docs, 
    embedding=embedding, 
    persist_directory="../chroma_langchain_db", 
    collection_name="trading_data")

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o-mini")
vprompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("what are the available stocks data for bmri?"))