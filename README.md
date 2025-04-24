## SimpleRAG

一个简单的 RAG 系统

#### VDB

用 Milvus 做后台 vdb，milvus 启动命令：

```shell
sudo docker compose -f docker-compose-milvus.yaml up -d
```

不用的时候记得关：

```shell
sudo docker compose -f docker-compose-milvus.yaml stop
```

embedding func 用的也是 Milvus 内置的 embedding

#### rerank

模仿网易 Qanything 的 rerank 代码。不需要配置什么环境，模型路径：(s31) /home/ysy/code/simple-RAG/Mistral-7B-Instruct-v0.2



#### LLM

直接用 lmcache 进行推理。

```shell
# requires python >= 3.10 and nvcc >= 12.1
pip install lmcache lmcache_vllm
```

