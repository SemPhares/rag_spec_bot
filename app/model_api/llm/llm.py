from enum import Enum
from pydantic import BaseModel, Field
from typing import List


__all__ = ["LlmModelName", "LlmInput", "LlmOutput"]


class LlmModelName(str, Enum):
    CUSTOM_LLAMA2 = "cgi_specbot_llama2"
    CUSTOM_MISTRAL = "cgi_specbot_mistral"
    LLAMA2 = "llama2"
    MISTRAL = "mistral"


class LlmInput(BaseModel):
    context: str = Field(title="context", example="The IT documentation uses Oracle Database", description="Some complementary information", default="")
    question: str = Field(title="question", example="What are you ?", description="The question sent to the LLM engine")
    doc_ids: List[str] = Field(title="document identifiers", example=[], description="The list of all the documents to use to answer a question")


class LlmOutput(BaseModel):
    response: str
