from abc import ABC, abstractmethod

class LLM(ABC):
  @abstractmethod
  def generate(self, prompts: list[str], **kwargs):
    """LLM generate function."""
    return NotImplemented