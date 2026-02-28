import torch
from miditok import REMI, TokenizerConfig
from symusic import Score
from music_transformer import MusicTransformer
from pathlib import Path
import torch.nn.functional as F

config = TokenizerConfig(
    num_velocities=16,
    use_chords=True,
    use_programs = False
)
tokenizer = REMI(config)

device = torch.device('cpu')
model = MusicTransformer(vocab_size=len(tokenizer)).to(device)
checkpoint = torch.load('best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

generated = [tokenizer.vocab['BOS_None']]

for i in range(500):
    x = torch.tensor([generated]).to(device)
    
    if x.size(1) > 511:
        x = x[:, -511:]

    with torch.no_grad():
        logits = model(x)

        next_token_logits = logits[0, -1, :] / 1.0

        probs = F.softmax(next_token_logits, dim = -1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

    if i % 50 == 0:
        print(f'Generated {i} / 500')


midi_output = tokenizer.decode([generated])
midi_output.dump_midi('generated_midi.mid') 