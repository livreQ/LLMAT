import torch
from torch import nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from src.utils.configs import LLMFeatureType


class GPT2(nn.Module):

    def __init__(self, kind, reduction=LLMFeatureType.LAST_TOKEN):

        assert kind in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        super().__init__()
        self.reduction = reduction
        self.pretrain_model = kind
        self.config = GPT2Config.from_pretrained(self.pretrain_model)
        self.config.embd_pdrop = 0
        self.config.hidden_dropout = 0
        self.config.attn_pdrop = 0
        self.config.resid_dropout = 0
        self.feature_extractor = GPT2Model.from_pretrained(self.pretrain_model, config=self.config)
        self.feature_dim = self.feature_extractor.config.n_embd
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        return GPT2Tokenizer.from_pretrained(self.pretrain_model)

    def forward(self, data):
        feat = self.forward_features(data)
        return feat

    def forward_features(self, data):
        input_ids = data["input_ids"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
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
