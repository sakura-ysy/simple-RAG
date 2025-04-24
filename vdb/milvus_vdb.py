from vdb.abstract_vdb import VDB
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from pymilvus import MilvusClient, model, DataType

class MilvusVDB(VDB):
    def __init__(self, config: dict):
        super().__init__()
        milvus_uri = config.get("uri", "http://localhost:19540")
        self.emb_fn = model.DefaultEmbeddingFunction()
        self.client = MilvusClient(
            uri=milvus_uri,
        )

        use_old = config.get("use_old", True)
        self.collection_name = config.get("collection_name", "ysy_collection")
        if self.client.has_collection(self.collection_name):
            if use_old:
                return
            else:
                self.client.drop_collection(self.collection_name)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector", # Name of the vector field to be indexed
            index_type="HNSW", # Type of the index to create
            index_name="vector_index", # Name of the index to create
            metric_type="L2", # Metric type used to measure similarity
            params={
              "M": 64, # Maximum number of neighbors each node can connect to in the graph
              "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
            } # Index building params
        )
        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="id", # Name of the ID field
            datatype=DataType.INT64, # Type of the field
            is_primary=True, # Whether this field is a primary key
            auto_id=True, # Whether to auto-generate IDs for this field
        )

        schema.add_field(
            field_name="vector", # Name of the vector field
            datatype=DataType.FLOAT_VECTOR, # Type of the field
            dim=768, # Dimension of the vectors
        )
        schema.add_field(
            field_name="text", # Name of the text field
            datatype=DataType.VARCHAR, # Type of the field
            max_length=1024, # Maximum length of the text
        )

        self.client.create_collection(
          collection_name=self.collection_name,
          schema=schema,
          index_params=index_params,
        )
        self.client.load_collection(self.collection_name)

    def insert_file(self, file_path: str):
        """Insert to vdb from file."""
        if not os.path.exists(file_path):
          raise FileNotFoundError(f"File {file_path} does not exist.")
        
        # Load documents from a local directory
        loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path))
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        texts = [doc.page_content for doc in docs]
        vectors = self.emb_fn.encode_documents(texts)
        
        data = [{"vector": vector, "text": texts[index]} for (index, vector) in enumerate(vectors)] 
        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

    def insert_chunks(self, chunks: list[str]):
        """Insert to vdb from raw chunks."""
        if not chunks:
          raise ValueError("Chunks list is empty.")
        
        vectors = self.emb_fn.encode_documents(chunks)
        data = [{"vector": vector, "text": chunks[index]} for (index, vector) in enumerate(vectors)] 
        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        self.client.flush([self.collection_name])

    def retrieve(self, query, top_k=3):
        vectors = self.emb_fn.encode_documents([query])
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=top_k,  # Return top 3 results
            output_fields=["text"],  # Return the text field
        )

        return search_res

