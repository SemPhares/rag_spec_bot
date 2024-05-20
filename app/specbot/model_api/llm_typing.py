from enum import Enum
from pydantic import BaseModel, Field
from typing import List


__all__ = ["LlmModelName", "LlmInput", "LlmOutput"]


class llm_name(Enum):
    mistral = "mistral"
    llama3 = "llama3"
    phi = "phi"


class llm_input(BaseModel):
    context: str = Field(title="context", example="The IT documentation uses Oracle Database", description="Some complementary information", default="")
    query: str = Field(title="question", example="What are you ?", description="The question sent to the LLM engine")
    doc_ids: List[str] = Field(title="document identifiers", example=[], description="The list of all the documents to use to answer a question")


class llm_output(BaseModel):
    response: str
