from abc import ABC, abstractmethod

class VDB(ABC):
    @abstractmethod
    def insert_file(self, file_path: str):
        """Insert to vdb from file."""
        return NotImplemented

    def insert_chunks(self, chunks: list[str]):
        """Insert to vdb from raw chunks."""
        return NotImplemented

    @abstractmethod
    def retrieve(self, query):
        """Retrieve some context by query from the vdb."""
        return NotImplemented
