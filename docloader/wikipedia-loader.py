from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

query = "aplikasi pintu"
wikipedia = WikipediaLoader(query=query, load_max_docs=4, lang="id")
documents = wikipedia.load()
print(len(documents))
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    { "context": retriever, "question": RunnablePassthrough()  } | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("what is pintu crypto? Who is the founder and what are their products? explain in english"))
