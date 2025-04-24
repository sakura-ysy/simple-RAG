import asyncio
from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from langchain_core.documents import Document
import concurrent.futures
from abc import ABC, abstractmethod


class RerankBackend(ABC):
    def __init__(self, config: dict):
        self.use_cpu = config["rerank"].get("use_cpu", False)
        self.model = config["rerank"].get("rerank_model")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.spe_id = self._tokenizer.sep_token_id
        self.overlap_tokens = 80
        self.batch_size = config["rerank"].get("batch_size", 4)
        self.max_length = config["rerank"].get("max_length", 512)
        self.return_tensors = None
        self.workers = config["rerank"].get("thread", 4)

    @abstractmethod
    def inference(self, batch) -> List:
        pass

    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)

        # 在 chunk1 的末尾添加分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为分隔符添加 attention mask

        # 添加 chunk2 的内容
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])

        # 在整个序列的末尾再添加一个分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为最后的分隔符添加 attention mask

        if 'token_type_ids' in chunk1:
            # 为 chunk2 和两个分隔符添加 token_type_ids
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)

        return chunk1

    def tokenize_preproc(self,
                         query: str,
                         passages: List[str],
                         ):
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 2  # 减2是因为添加了两个分隔符

        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self._tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                         add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                merge_inputs_idxs.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    async def get_rerank(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self._tokenizer.pad(
                    tot_batches[k:k + self.batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                future = executor.submit(self.inference, batch)
                futures.append(future)
            # debug_logger.info(f'rerank number: {len(futures)}')
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        # print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores

    async def arerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """Embed search docs using async calls, maintaining the original order."""
        batch_size = self.batch_size  # 增大客户端批处理大小
        all_scores = [0 for _ in range(len(source_documents))]
        passages = [doc.page_content for doc in source_documents]

        tasks = []
        for i in range(0, len(passages), batch_size):
            task = asyncio.create_task(self.get_rerank(query, passages[i:i + batch_size]))
            tasks.append((i, task))

        for start_index, task in tasks:
            res = await task
            if res is None:
                return source_documents
            all_scores[start_index:start_index + batch_size] = res

        for idx, score in enumerate(all_scores):
            source_documents[idx].metadata['score'] = round(float(score), 2)
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents