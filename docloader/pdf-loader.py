from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
query = "aplikasi pintu"
pdf = PyPDFLoader(file_path="../proposal-kegiatan-itcc-2024.pdf")
documents = pdf.load()
splitted_docs = text_splitter.split_documents(documents)

print(len(documents))
print(len(splitted_docs))

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    { "context": retriever, "question": RunnablePassthrough()  } | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("what is the TOTAL RAB KOTOR ITCC 2024 for the event?"))
