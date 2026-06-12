import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cuda.is_built())
import requests
from model import TransformerLanguageModel

# --- Configuration & Device Setup ---
print(f"Is CUDA available? {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64 
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 3

print(f"Running training on: {device}")

# --- Data Loading ---
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    # CRITICAL: Move data to GPU here
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Initialization ---
model = TransformerLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer)
model = model.to(device) # Move entire model to GPU

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Training Loop ---
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get batch (already moved to device inside get_batch)
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining Complete.")