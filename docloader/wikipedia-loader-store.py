
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

query = "aplikasi pintu"
wikipedia = WikipediaLoader(query=query, load_max_docs=4, lang="id")
documents = wikipedia.load()

textsplitting = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_doc = textsplitting.split_documents(documents)

# print(splitted_doc)

vectorstore = Chroma.from_documents(
    documents=splitted_doc, 
    embedding=embedding,
    collection_name="wikipedia_aplikasi_pintu",
    persist_directory="../chroma_langchain_db"
    )

# print(len(documents))
# from langchain_chroma import Chroma
# vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
# retriever = vectorstore.as_retriever()
