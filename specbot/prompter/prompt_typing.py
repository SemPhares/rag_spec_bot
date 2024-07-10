from pydantic import BaseModel, Field
from typing import Union

class prompt_input(BaseModel):
   query: str = Field(title="question", description="User query")
   retrieved_chunks: Union[list, str] = Field(title="doc_retrieved", description="The document retrieved")

class classification_prompt_input(BaseModel):
   query: str = Field(title="question", description="User query")
   categories: Union[list, str] = Field(title="categories", description="The categories to be used for classification")

class prompt_output(BaseModel):
   prompt: str = Field(title="prompt", description="The prompt to be used for the model")