from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# not working
json_schema = {
    "title": "jokeb",
    "description": "Use this structure for joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}


structured_llm = llm.with_structured_output(json_schema)

result = structured_llm.invoke("Tell me a joke about suck manager")

for chunk in result:
    print(chunk, end="", flush=True)
