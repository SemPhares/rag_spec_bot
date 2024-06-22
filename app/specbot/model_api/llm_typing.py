from pydantic import BaseModel, Field


class llm_input(BaseModel):
    model_name : str = Field(title="model_name", description="The name of the model to use for the response")
    input: str = Field(title="question", description="The question sent to the LLM engine")


class llm_output(BaseModel):
    model_name : str = Field(title="modelname", description="The name of the model used to generate the response")
    response: str

class llam_cpp_local_input(llm_input):
   model_path: str = Field(title="model_path", description="The path to the model to use for the response")

class llama_cpp_petrained_input(llm_input):
   repo_id: str = Field(title="repo_id", description="The repo id of the model to use for the response")
   filename : str = Field(title="filename", description="The filename of the model to use for the response")