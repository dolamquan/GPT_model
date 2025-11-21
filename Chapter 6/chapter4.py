import tiktoken
import torch
import torch.nn as nn



GPT_CONFIG_124 = {
    "vocab_size": 50257, # Vocabulary size of the model
    "context_length": 1024, # Maximum context length for input sequences
    "embed_dim": 768, # Dimensionality of the token embeddings
    "n_heads": 12, # Number of attention heads in the multi-head attention mechanism
    "n_layers": 12, # Number of transformer layers in the model
    "dropout": 0.1, # Dropout rate for regularization
    "qkv_bias": False, # Whether to include bias terms in the query, key, value projections
}





# OpenAI's tokenizer library - It converts text to token IDs and back
tokenizer = tiktoken.get_encoding("gpt2") # Load GPT-2 tokenizer





# Stabilizes training by keeping activations in a consistent range
# Transformers use it heavily:
# Before self-attention and feedforward layers
# Or after these layers, depending on the architecture
# Keeps activations stable by normalizing each token's fetures (embedding vector) so they have mean 0 and variance 1
# This helps gradients flow better and prevents exploding/vanishing gradients
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # Small constant to prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Learnable scaling parameter - initialized to ones
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift
    

# Activation function used in transformers
# Smooth approximation of ReLU
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


# Use GELU function to implement a feed forward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Creates a sequence of layers that make up the feed forward block
        self.layers = nn.Sequential(
            nn.Linear(cfg['embed_dim'], cfg['embed_dim'] * 4),
            GELU(),
            nn.Linear(cfg['embed_dim'] * 4, cfg['embed_dim']),
        )
    
    # Passsses the input through the feed forward layers
    def forward(self, x):
        return self.layers(x)


# Multi-head self-attention mechanism
# Allows the model to focus on different parts of the input sequence simultaneously
class MultiHeadAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias =False):
        # Calls the constructor of the parent class nn.Module
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Dimension of each attention head -> features per head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Contain the combined results from all attention heads
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values

        context_vecs = context_vecs.transpose(1,2).contiguous().view(b, num_tokens, self.d_out)
        output = self.out_proj(context_vecs)
        return output


# Each block handles self-attention, feed forward layers, layer normalization, dropout, and residual connections
# MLP: Multi-Layer Perceptron - a small feed-forward neural network: A stack of fully connected
# linear layers with non linear activations in between

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        # Creates a multi-head attention layer that allows the model to "look at" different parts of the input sequence simultaneously
        # All parameters are derived from the cfg dictionary
        # It outputs representations of tokens that have contextual meaning
        self.att = MultiHeadAttention_v2(
            d_in=cfg['embed_dim'],
            d_out=cfg['embed_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['dropout'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )

        # Feed forward and Normalization layers
        self.ff = FeedForward(cfg) # A simple MLP applied to each token after attention
        self.norm1 = LayerNorm(cfg['embed_dim']) # Normalizes features to stabilize training
        self.norm2 = LayerNorm(cfg['embed_dim'])
        self.dropout = nn.Dropout(cfg['dropout']) # Regularization to prevent overfitting
    
    def forward(self, x):

        # Self-attention sub-layer with residual connection
        # 1. Save the input for the residual connection
        # 2. Normalaize it -> pass it through attention to get attention weights
        # 3. Apply dropout for regularization
        # 4. Add the original input (shortcut) back to the output of attention

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['embed_dim'])
        self.drop = nn.Dropout(cfg['dropout'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['embed_dim'])
        self.out_head = nn.Linear(cfg['embed_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_length = in_idx.shape

        tok_embeddings = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_length, device=in_idx.device))

        x = self.drop(tok_embeddings + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# Model: GPTModel
# idx: input token IDs
# max_new_tokens: number of new tokens to generate
# context_size: maximum context size for the model

def generate_text(model,idx,max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Crop context if needed

        # Feed the current sequence to the model to get predictions for the next token
        # torch.no_grad(): disables gradient calculation to save memory and computation during inference
        with torch.no_grad():
            logits = model(idx_cond)
    
        # We only care about the logits of the last token position, because that's the model's prediction for the next token
        logits = logits[:, -1, :]  # Focus on the last token's logits
        probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        next_token = torch.multinomial(probabilities, num_samples=1)  # Sample the next token -> can use torch.argmax 
        idx = torch.cat((idx, next_token), dim=1)  # Append the sampled token to the sequence
    return idx