from pydantic import BaseModel, Field
from typing import List, Optional, Union
from langchain_core.documents import Document
from config.global_config import GlobalConfig


class evaluation_input(BaseModel):

   user_query: str = Field(title="question", description="The question sent to the LLM engine")
   actual_output: str = Field(title="response", description="The response from the LLM engine")
   retrieval_context: list[Document] = None
   file_name: list = None
   generation_time: float = None
   

class base_evaluation_input(BaseModel):
   CHUNCK_SIZE: int = GlobalConfig.CHUNCK_SIZE
   CHUNK_OVERLAP: int = GlobalConfig.CHUNK_OVERLAP
   TEMPERATURE: float = GlobalConfig.TEMPERATURE
   CONTEXT_WINDOW: int = GlobalConfig.CONTEXT_WINDOW
   CONSERVATIVE_TEMPERATURE: float = GlobalConfig.CONSERVATIVE_TEMPERATURE
   RETRIEVER_TOP_K: int = GlobalConfig.RETRIEVER_TOP_K
   RETRIEVER_SCORE_THRESHOLD: float = GlobalConfig.RETRIEVER_SCORE_THRESHOLD
   CURRENT_TIME: str = ""


class evaluation_output(base_evaluation_input, evaluation_input):
   pass


###@ copié collé de deepeval/evaluate
class MetricMetaDonnees(BaseModel):
   metric: str 
   threshold: float
   success: bool
   score: Optional[float] = None
   reason: Optional[str] = None
   strict_mode: Optional[bool] = Field(False, alias="strictMode")
   evaluation_model: Optional[str] = Field(None, alias="evaluationModel")
   error: Optional[str] = None
   evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")


class Result(BaseModel):
   input: str
   success: bool
   metrics_metadata: list[MetricMetaDonnees]
   actual_output: str
   expected_output: str
   context: list[str]
   retrieval_context: list[str]



