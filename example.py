import yaml
import asyncio
from langchain_core.documents import Document
from vdb import CreateVDBInstance
from llm import CreateLLMInstance

from rerank.rerank_torch_backend import RerankTorchBackend
from utils import combine_input_prompt_chunks

config = {}
config_path = "config/config1.yaml"
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

vdb = CreateVDBInstance(config)
# llm = CreateLLMInstance(config)

# insert some chunks to vdb through local file
file_path = "documents/doc1.txt"
vdb.insert_file(file_path)

# start query
query = "Give me a summary of LLM"
# retrieve
results = vdb.retrieve(query, top_k=5)
# rerank
source_documents = [Document(result["entity"]["text"]) for result in results[0]]

rerank_backend = RerankTorchBackend(config)
source_documents = asyncio.run(rerank_backend.arerank_documents(query, source_documents))

sys_prompt = "You are a helpful assistant. Answer the question based on the context provided."
user_prompt = [sys_prompt] + [chunk for chunk in source_documents] + [query]
user_prompt = combine_input_prompt_chunks(user_prompt)
# llm_response = llm.generate([user_prompt])
# print("llm_response:", llm_response)












