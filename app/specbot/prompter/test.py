from prompt import build_rag_prompt, build_sum_prompt
from langchain_core.documents import Document

test = [Document(page_content="text",
        metadata={"version": "version", "book": "book"}),
        Document(page_content="text",
        metadata={"version": "version", "book": "book"})]

print(build_rag_prompt('test_query',test))

print(build_sum_prompt(test))