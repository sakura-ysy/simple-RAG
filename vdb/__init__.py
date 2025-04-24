from vdb.abstract_vdb import VDB
from vdb.milvus_vdb import MilvusVDB

def CreateVDBInstance(config: dict) -> VDB:
    # Replace 'cuda' with 'cuda:<device id>'
    vdb_type = config.get("vdb", "milvus")
    match vdb_type:
        case "milvus":
            return MilvusVDB(config["milvus"])
        case "faiss":
            return NotImplementedError("FAISS backend is not implemented")
        case _:
            raise ValueError(f"Invalid vdb type: {vdb_type}")