from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from typing import Optional
from pydantic import BaseModel, Field


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about suck manager")

for chunk in result:
    print(chunk, end="", flush=True)


print(result.setup)