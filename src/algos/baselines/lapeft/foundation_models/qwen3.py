import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from src.utils.configs import LLMFeatureType


MODEL_MAP = {
    "qwen3-8b": "Qwen/Qwen3-8B",
}


def get_qwen3_tokenizer(kind):
    model_name = MODEL_MAP[kind]
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


class Qwen3Regressor(nn.Module):

    def __init__(self,\
                 kind,
                 tokenizer,
                 reduction=LLMFeatureType.LAST_TOKEN,
                 n_last_hidden_units=100):

        assert kind in MODEL_MAP
        super().__init__()
        self.tokenizer = tokenizer
        self.reduction = reduction
        model_name = MODEL_MAP[kind]
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.attention_dropout = 0.0
        self.feature_extractor = AutoModel.from_pretrained(
            model_name, config=self.config, trust_remote_code=True
        )
        self.feature_dim = self.config.hidden_size
        self.head = nn.Sequential(nn.Linear(self.feature_dim, n_last_hidden_units),\
                                  nn.ReLU(),\
                                  nn.Linear(n_last_hidden_units, 1))

    def forward(self, data):
        feat = self.forward_features(data)
        return self.head(feat)

    def forward_features(self, data):
        input_ids = data["input_ids"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = data.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
            feat = self.feature_extractor(input_ids, attention_mask=attention_mask)[0]
        else:
            feat = self.feature_extractor(input_ids)[0]

        if self.reduction == LLMFeatureType.FIRST_TOKEN:
            feat = feat[:, 0, :]
        elif self.reduction == LLMFeatureType.LAST_TOKEN:
            # Find the last token position (before padding)
            sequence_lengths = (torch.eq(input_ids, self.tokenizer.pad_token_id)\
                                                             .long().argmax(-1) - 1).to(device)
            feat = feat[torch.arange(feat.shape[0], device=device), sequence_lengths]  # (batch_size, feature_dim)
        elif self.reduction == LLMFeatureType.AVERAGE:
            # Masking---0 for everything after <eos> and 1 otherwise
            mask = ~input_ids.eq(self.tokenizer.pad_token_id).to(feat.device)
            mask = mask[:, :, None].float()  # (batch_size, seq_len, 1)
            #  (batch_size, 1, hidden_size) / (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
            feat = (feat * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)

        return feat.squeeze(1)

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
