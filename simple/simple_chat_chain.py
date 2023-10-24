from dotenv import load_dotenv

load_dotenv()

from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

messages = [
    SystemMessage("you are a mathematician you calculate number of character in a sentence"),
    HumanMessage("How many characters in this sentence?")
]

chain = model | parser
result = chain.invoke(messages)
print(result)