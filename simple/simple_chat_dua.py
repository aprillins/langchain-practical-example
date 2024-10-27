from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini")
# non streaming
# result = model.invoke([
#     SystemMessage("You are a sarcastic person who always gives response less than 30 words"),
#     HumanMessage("who are you?")
#     ])
# print(result.content)

# streaming method
result = model.stream([
    SystemMessage("You are a sarcastic person who always gives response always 30 words"),
    HumanMessage("who are you?")
    ])

for chunk in result:
    print(chunk.content, end="", flush=True)