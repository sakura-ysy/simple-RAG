from transformers import AutoModelForSequenceClassification
from rerank.rerank_backend import RerankBackend
import torch


class RerankTorchBackend(RerankBackend):
    def __init__(self, config: dict):
        super().__init__(config)
        self.return_tensors = "pt"
        self.model_path = config["rerank"].get("rerank_model")
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                                                                         return_dict=False)
        if self.use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        self._model = self._model.to(self.device)
        print("rerank device:", self.device)

    def inference(self, batch):
        # 准备输入数据
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        # 执行推理 输出为logits
        result = self._model(**inputs, return_dict=True)
        sigmoid_scores = torch.sigmoid(result.logits.view(-1, )).cpu().detach().numpy()

        return sigmoid_scores.tolist()