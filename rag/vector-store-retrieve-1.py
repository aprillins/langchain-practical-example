from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    )
]

vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-3-small"))

# search_result = vectorstore.similarity_search("rabbit")
# print(search_result)

# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)

# OR

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# retriever.batch(["cat", "shark"])


# print(retriever.batch(["cat", "shark"]))

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# prompt = ChatPromptTemplate.from_messages([("human", message)])
prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(message)])

llm = ChatOpenAI(model="gpt-4o-mini")
rag_chain = { "context": retriever, "question": RunnablePassthrough() } | prompt | llm
response = rag_chain.invoke("tell me about shark")
print(response.content)



