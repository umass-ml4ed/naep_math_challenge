import os
import torch
from torch import nn
from transformers import T5EncoderModel
from typing import Optional


class FlanT5encoder(nn.Module):
    def __init__(self, lm, num_label):
        super().__init__()
        self.flan_t5_encoder = T5EncoderModel.from_pretrained(lm)
        self.linear = nn.Linear(1024, num_label)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.FloatTensor] = None, ):
        hidden_states = self.flan_t5_encoder(input_ids, attention_mask)
        output = torch.div(
            torch.sum(torch.mul(hidden_states.last_hidden_state, attention_mask.unsqueeze(-1).float()), dim=1),
            torch.sum(attention_mask, dim=-1).unsqueeze(-1).float())
        logits = self.linear(output)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss, logits



