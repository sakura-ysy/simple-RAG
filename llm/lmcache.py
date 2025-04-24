from llm.abstract_llm import LLM
import lmcache_vllm
import torch
from lmcache_vllm.blend_adapter import (append_separator,
                                        combine_input_prompt_chunks)
from lmcache_vllm.vllm import LLM, SamplingParams

class LMCache(LLM):

    def __init__(self, config: dict):
        """Initialize the LMCache with a configuration."""
        self.model = config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.5)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.max_tokens = config.get("max_tokens", 30)

        self.llm = LLM(model=self.model, gpu_memory_utilization=self.gpu_memory_utilization, tensor_parallel_size=1, dtype="half")

        self.sampling_params = SamplingParams(temperature=0.0,
                                                    top_p=0.95,
                                                    max_tokens=self.max_tokens)

    def generate(self, prompts: list[str], **kwargs):
        """Generate responses from the LLM with caching."""
        return self.llm.generate(prompts, self.sampling_params)
