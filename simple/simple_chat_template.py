from dotenv import load_dotenv

load_dotenv()

from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

system_template = "describe the following animal concisely and maximum {number} words"

messages = [
    ("system", system_template),
    ("user", "{text}")
]

messages_dua = [
    SystemMessage(system_template),
    HumanMessage(content="{text}"),
]

prompt_template = ChatPromptTemplate.from_messages(
    messages
)

prompt = prompt_template.invoke({"number": 30, "text": "elephant"})
result = prompt.to_messages()
output = model.invoke(result)
print(parser.invoke(output))

# Chaining method
# chain = prompt_template | model | parser
# result = chain.invoke({"number": 100, "text": "dog"})
# print(result)