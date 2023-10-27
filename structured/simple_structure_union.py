
from langchain_openai import ChatOpenAI
from typing import Union, Optional
from pydantic import BaseModel, Field


llm = ChatOpenAI(model="gpt-4o-mini")

# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")

class StockAnalysis(BaseModel):
    """Analyze stock"""
    stock_code: str = Field(description="The stock name in discussion")
    country: str = Field(description="Country of the stock")
    analysis: str = Field(description="A simple 50 words analysis about the stock condition")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse, StockAnalysis]


structured_llm = llm.with_structured_output(FinalResponse)

result = structured_llm.invoke("Do you know about BMRI?")

for chunk in result:
    print(chunk, end="", flush=True)
