from torch import nn as nn
import torch
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = nn.Embedding(512, d_model)
        
        self.dropout = nn.Dropout(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
                d_model = d_model,
                nhead = 4,
                dim_feedforward = 256,
                batch_first = True,
                dropout = 0.3)

        self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers = 2)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)

        positions = torch.arange(seq_len, device = x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        
        x = self.dropout(x)

        x = self.transformer(x)

        logits = self.fc(x)

        return logits


        