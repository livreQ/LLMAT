from torch import nn
from transformers import AutoModel, AutoTokenizer


class MolFormer(nn.Module):

    def __init__(self, kind=None):
        super().__init__()
        self.kind = kind
        # if kind == "molformer":
        self.pretrain_model = "ibm/MoLFormer-XL-both-10pct"
        self.feature_extractor = AutoModel.from_pretrained(
            self.pretrain_model,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.feature_dim = self.feature_extractor.config.hidden_size
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrain_model, trust_remote_code=True, use_fast=False)

    def forward(self, data):
        feat = self.forward_features(data)
        return feat

    def forward_features(self, data):
        input_ids, attn_mask = data["input_ids"], data["attention_mask"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids, attn_mask).pooler_output
        return feat

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
