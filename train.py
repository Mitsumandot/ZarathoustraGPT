import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken 

context_length = 256
batch_size = 64
n_embd = 384
eval_interval = 10
learning_rate = 3e-4
eval_iters = 200
dropout = 0.2
n_head = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

with open('archive/nietzsche.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# gpt2 encoding, but makes the training impossible for a single machine as the size of the embeddings becomes massive
#enc = tiktoken.get_encoding("gpt2")
#vocab_size = enc.n_vocab

chars = sorted(list(set(text)))
vocab_size = len(chars)

#Encoding the dataset.
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#function that runs every eval_interval, gives the mean error of train and test data over eval_iters examples
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #initialize the matrix of embeddings of size vocab_size x n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #initialize the matrix that contains the position informations of each token in the context
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        #sequence of blocks, each one containing a self-attention layer and a feed forward layer, the LayerNorm normalizes the output (mean = 0 and var = 1)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=n_head),
            Block(n_embd, n_head=n_head),
            Block(n_embd, n_head=n_head),
            Block(n_embd, n_head=n_head),
            Block(n_embd, n_head=n_head),
            Block(n_embd, n_head=n_head),
            nn.LayerNorm(n_embd)
        )
        #linear layer that outputs vectors of size vocab_size, the purpose is to produce a probability later using softmax on every token of the vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # get the embeddings of each token in the current batch
        tok_emb = self.token_embedding_table(idx)
        #get positional embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        #add the positional embedding
        final_emb = tok_emb + pos_emb
        #forward through transformer
        final_emb = self.blocks(final_emb)
        #get the output vectors 
        logits = self.lm_head(final_emb)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -context_length:]
            
            logits, loss = self(idx_cond)
            #get the last token, because its the one that contains all the information necessary to predict the next token
            logits = logits[:, -1, :]
            
            #forward through softmax
            probs = F.softmax(logits, dim=-1)
            #introduces randomness in the selection of the token 
            idx_next = torch.multinomial(probs, num_samples=1)
            #add token to the sentence
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx
    
class Block(nn.Module):
    
    def __init__(self, n_embd, n_head): 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
#attention head
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        #key matrix containing the information about what exactly the value of the token contains
        self.key = nn.Linear(n_embd, head_size, bias=False)
        #define what the token is searching for in previous tokens
        self.query = nn.Linear(n_embd, head_size, bias=False)
        #define the actual information held by the token
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #each iteration, dropping out some neurones for regularization
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        #dot product between all key and queries
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #output embeddings by multiplying the the dotproducts matrix and the value matrix
        out = wei @ v
        return out


#creating parallel attention heads that combines their output into one embedding
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


#feed forward layer
class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), 
                                nn.ReLU(),
                                nn.Linear(4 * n_embd, n_embd),
                                nn.Dropout(dropout))
    
    def forward(self, x):
        return self.net(x)

#training & testing & saving
if __name__ == "__main__":
    

    model = BigramLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(5000):
        
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
        
        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), "./model/model2.pt")

    print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=400)[0].tolist()))
