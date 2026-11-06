import torch
from torch import nn
from transformers import RobertaModel, AutoConfig, RobertaTokenizer
from src.utils.configs import LLMFeatureType


class Roberta(nn.Module):
    def __init__(self,\
                 kind,
                 reduction=LLMFeatureType.FIRST_TOKEN):

        assert kind in ["roberta-base", "roberta-large"]
        super().__init__()

        self.reduction = reduction
        self.pretrained_model = kind
        self.config = AutoConfig.from_pretrained(self.pretrained_model)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.feature_extractor = RobertaModel.from_pretrained(self.pretrained_model, config=self.config)
        self.feature_dim = self.feature_extractor.config.hidden_size
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.pretrain_model)

    def forward(self, data):
        feat = self.forward_features(data)
        return feat

    def forward_features(self, data):
        input_ids, attention_mask = data["input_ids"], data["attention_mask"]
        device = next(self.parameters()).device
        input_ids, attention_mask = (
            input_ids.to(device, non_blocking=True),
            attention_mask.to(device, non_blocking=True),
        )
        out = self.feature_extractor(input_ids, attention_mask)
        feat = out[0]

        if self.reduction == LLMFeatureType.FIRST_TOKEN:
            feat = feat[:, 0, :]
        elif self.reduction == LLMFeatureType.LAST_TOKEN:
            # Find the last token position (before padding)
            eos_mask = input_ids.eq(self.tokenizer.eos_token_id).to(feat.device)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            batch_size, _, hidden_size = feat.shape
            feat = feat[eos_mask, :]  # (batch_size, hidden_size)
            feat = feat.view(batch_size, -1, hidden_size)[:, -1, :]
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
