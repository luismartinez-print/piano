import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import miditok
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
import pathlib
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import random
from symusic import Score

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

        positions = torch.arange(seq_len).unsqueeze(0).to(device)

        x = self.embedding(x) + self.pos_embedding(positions)
        
        x = self.dropout(x)

        x = self.transformer(x)

        logits = self.fc(x)

        return logits


for m_path in midi_files:
    try:
        
        original_score = Score(m_path)
        tokens_obj = tokenizer.encode(original_score)
        tokens = tokens_obj[0].ids
        
        for i in range(0, len(tokens) - max_len, stride):
            chunk = tokens[i : i + max_len]
            all_chunks.append({
                "input_ids": torch.tensor(chunk, dtype=torch.long)
            })
            
        if len(all_chunks) > 100000:
            break
                
    except Exception as e:
        continue

print(f"Created {len(all_chunks)} sequences.")

random.seed(69)
random.shuffle(all_chunks)

train_size = int(0.9 * len(all_chunks)) 
train_data = all_chunks[:train_size]
val_data = all_chunks[train_size:]

print(f"Train: {len(train_data)} | Val: {len(val_data)}")


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

print(f"Training on {len(train_loader)} samples, Validating on {len(val_loader)} samples.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicTransformer(vocab_size=len(tokenizer)).to(device)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicTransformer(vocab_size = len(tokenizer)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay = 0.01)

epochs = 50
patience = 7
best_val_loss = float('inf')
counter = 0
check_path = 'best_model.pt'

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        if batch is None:
          continue
        tokens = batch['input_ids'].to(device)
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()


    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            val_token = batch['input_ids'].to(device)
            x_val = val_token[:, :-1]
            y_val = val_token[:, 1:]

            logits_val = model(x_val)
            v_loss = criterion(logits_val.reshape(-1, logits_val.size(-1)), y_val.reshape(-1))

            total_val_loss += v_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f'{epoch +1}, Train loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_val_loss},
                check_path)
        else:
            counter += 1
            if counter >= patience:
              print('early stopping')
              break


