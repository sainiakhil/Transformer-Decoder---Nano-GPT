import torch 
import torch.nn as nn
import torch.nn.functional as F



# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel? 
block_size = 256  # what is the maximum context length for predictions? 
n_embd = 384 # the dimensionality of the character embedding vectors
learning_rate = 3e-3
max_iters = 5000
eval_interval = 500  
eval_iters = 200
vocab_size = None
loss = None
dropout = 0.2
n_head = 6 # number of heads in multi-head attention 
n_layers = 6 # number of transformer blocks 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------------------------------------------------

torch.manual_seed(1337)

# Importing the tiny shakespeare dataset
# !wget "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with open('Text Dataset.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()


# here are all the unique character in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,head_size)
    q = self.query(x) # (B,T,head_size)

    # compute attention scores
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T) -> explore this scaling factor
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei)

    v = self.value(x) # (B,T,head_size)
    out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
    return out
  
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      out = self.dropout(out)
      return out


class FeedForward(nn.Module):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd), # expansion factor of 4 cause in transformer paper it is mentioned that the inner layer has 4 times more features like 512 and 2048
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self, x):
      return self.net(x)
  

class Transformer_Block(nn.Module): #transformer decoder block without cross attention
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like to have 
    super().__init__()
    head_size = n_embd//n_head
    self.sa_head = MultiHeadAttention(n_head, head_size) # 4 heads of self-attention and head size = 32//4 = 8
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
      x = x + self.sa_head(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))

      return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
      super().__init__()
      # each token directly reads off the logits for the next token from a lookup table
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)
      self.blocks = nn.Sequential(*[Transformer_Block(n_embd, n_head) for _ in range(n_layers)])

      # self.blocks = nn.Sequential(

      #   Transformer_Block(n_embd, 4), # 4 heads of self-attention
      #   Transformer_Block(n_embd, 4),
      #   Transformer_Block(n_embd, 4),
      #   nn.LayerNorm(n_embd) 
      #)
      
      # self.sa_head = MultiHeadAttention(4, n_embd // 4) 
      # self.ff = FeedForward(n_embd)
      self.layer_norm = nn.LayerNorm(n_embd) # final layer norm
      self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
    
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_head(x) # (B,T,C)
        # x = self.ff(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.layer_norm(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
    
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # crop the idx to the last block size tokens
            idx_cond = idx[:, -block_size:] #---check it again to learn(what is going on at backend)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    


# -------------------------\/\/main code starts here\/\/-------------------------
#---------------------------------------------------------------------
#----------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
  """ take a string and return a list of integers """
  return [stoi[c] for c in s]

def decode(l):
  """ take a list of integers and return the corresponding string """
  return ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# ------------------------------------------------------------------
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
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


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# ------------------------------------------------------------------
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("final loss:", loss.item())
# ------------------------------------------------------------------


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))