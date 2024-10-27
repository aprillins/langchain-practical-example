from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """ multiple two numbers """
    return a * b

print(multiply.invoke({"a": 20, "b": 30}))