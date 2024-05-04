from fastapi import APIRouter, Query

from ...handler.v1 import llm
from ...model.llm import LlmModelName, LlmInput, LlmOutput


__all__ = ["router"]


router = APIRouter(
    prefix="/llm",
    tags=["llm"],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)


@router.post("/ask",
             summary="Ask a question",
             description="Ask a question to Ollama LLM service",
             response_model=LlmOutput)
def ask_llm(llm_input: LlmInput,
            model_name: LlmModelName = Query(default=LlmModelName.CUSTOM_MISTRAL)) -> LlmOutput:
    return llm.ask_question(model_name=model_name, llm_input=llm_input)
