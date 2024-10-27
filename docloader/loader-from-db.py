from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name="wikipedia_aplikasi_pintu",
    persist_directory="../chroma_langchain_db",
    embedding_function=embedding,
    )

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    { "context": retriever, "question": RunnablePassthrough()  } | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("what is pintu crypto? Who is the founder and what are their products? explain in english"))
