import torch
from typing import Optional


class GPTJWrapper(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.gptj_class = lm

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.FloatTensor] = None, ):
        loss, logits = self.gptj_class(input_ids, attention_mask, labels)
        if isinstance(logits, tuple):
            logits = logits[0]
        return loss, logits
