from llm.lmcache import LMCache 
from llm.abstract_llm import LLM

def CreateLLMInstance(config: dict) -> LLM:
    # Replace 'cuda' with 'cuda:<device id>'
    llm_type = config.get("llm", "lmcache")
    match llm_type:
        case "lmcache":
            return LMCache(config)
        case "vllm":
            return NotImplementedError("vllm backend is not implemented")
        case _:
            raise ValueError(f"Invalid llm type: {llm_type}")