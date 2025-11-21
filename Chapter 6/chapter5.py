from chapter4 import GPTModel, generate_text
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken


def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # add batch dim

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.view(-1).tolist()  # flatten to 1D list
    return tokenizer.decode(flat)

# Turns a long text file into a dataset of input-target pairs for GPT training
class GPTDatasetV1(torch.utils.data.Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, pad_id=50256):  # gpt2 endoftext
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # If too short, right-pad to produce one training example
        if len(token_ids) <= max_length:
            inp  = token_ids[:max_length]
            tgt  = token_ids[1:max_length+1]

            # pad to fixed length
            if len(inp) < max_length:
                inp = inp + [pad_id] * (max_length - len(inp))
            if len(tgt) < max_length:
                tgt = tgt + [pad_id] * (max_length - len(tgt))

            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))
            return

        # Normal sliding windows when long enough
        # Need i such that i+max_length+1 <= len(token_ids)
        for i in range(0, len(token_ids) - max_length, stride):
            inp = token_ids[i:i+max_length]
            tgt = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



# Function wraps the dataset inside a PyTorch DataLoader, which handles batching and shuffling
# It is to prepare batches of (input,target) pairs efficiently for model training
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    print("Dataset length:", len(dataset))  # helpful sanity check
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) # Move tensors to GPU if available
    target_batch = target_batch.to(device) # Move tensors to GPU if available

    logits = model(input_batch) # Logits are the raw,unnormalized scores output by the model

    logits_flat = logits.flatten(0,1)
    targets_flat = target_batch.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat) # Compute cross-entropy loss
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0 # Initialize total loss
    if len(data_loader) == 0: # Handle empty data loader
        return float('nan') # Return NaN if no data
    elif num_batches is None: # If num_batches not specified, use all batches
        num_batches = len(data_loader) 
    else: # Limit num_batches to available batches
        num_batches = min(num_batches, len(data_loader))
    
    # Iterate over batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches: # Process only up to num_batches
            # Calculate loss for the current batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Accumulate total loss
            total_loss += loss.item()
        else:
            break
    
    # Compute average loss over the processed batches
    avg_loss = total_loss / num_batches
    return avg_loss


# Measure model performance on training and validation sets
# model.eval() puts the model in evaluation mode -> disables dropout and other training-specific layers
# torch.no_grad() disables gradient calculation -> reduces memory consumption and speeds up computations

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Compute average loss on training and validation sets
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Switch back to training mode
    return train_loss, val_loss



def generate_and_print_sample(model, start_context, tokenizer, device, max_new_tokens=50):
    model.eval()  # Set model to evaluation mode
    context_size = model.pos_emb.weight.shape[0]  # Get context size from model
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # Encode start context/ prompt
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)  # Decode generated token IDs
    print("\nGenerated Sample Text:\n", decoded_text)
    model.train()  # Switch back to training mode


# model: the LLM to be trained
# train_loader: DataLoader for training data
# val_loader: DataLoader for validation data
# optimizer: Optimizer for model parameters
# device: Device to run the model on (CPU or GPU)
# num_epochs: Number of training epochs
# eval_freq: Frequency of evaluation during training (in steps)
# eval_iter: Number of batches to use for evaluation
# start_context: Initial text context for text generation
# tokenizer: Tokenizer for encoding and decoding text

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
                       ,start_context,tokenizer):
    
    train_losses = [] # Lists to store training and validation losses
    val_losses = [] # Lists to store training and validation losses
    track_tokens_seen =[] # Lists to track number of tokens seen during training

    tokens_seen = 0  # Initialize token counter
    global_step = -1  # Global step counter is used to track the number of optimization steps taken

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Calculate loss
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model parameters
            tokens_seen += input_batch.numel()  # Update token counter
            global_step += 1  # Increment global step counter

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,train_loader,val_loader,device,eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}:"
                      f" Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f},"
                      f" Tokens Seen = {tokens_seen}")
        
        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Last recorded Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")
        generate_and_print_sample(model, start_context, tokenizer, device)

    return train_losses, val_losses, track_tokens_seen
    



# Combine temperature sampling and top-k sampling for text generation

# idx: Initial token IDs (context)
# max_new_tokens: Number of new tokens to generate
# context_size: Maximum context size for the model / Maximum number of the tokens the model can "see" at once
# temperature: Temperature for scaling logits before sampling
# top_k: Number of top tokens to consider for top-k sampling
# eos_id: ID of the end-of-sequence token (optional)

def generate_text_modified(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Ensure context size limit
        with torch.no_grad(): # Call this because it is uneccessary during inference
            logits = model(idx_cond) # Get logits from the model
        
        logits = logits[:, -1, :]  # Focus on the last token's logits

        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k) # Returns the top-k largest elements
            min_val = top_logits[:, -1] # Get the smallest value among the top-k logits
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            next_token_id = torch.multinomial(probas, num_samples=1)  # Sample from the distribution
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy selection
            next_token_id = idx_next
        
        if next_token_id == eos_id:
            break  # Stop if end-of-sequence token is generated

        idx = torch.cat((idx, next_token_id), dim=1)  # Append the new token ID
    
    return idx







# Override random weights with the weights we downloaded from OpenAI

# Define helper function to assign weights
def assign(left,right):
    if left.shape!= right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


"""
    1. Load embedding weights
    2. Loops through every transformer block, and for each block:
        - loads Q,K,V projection weights and biases
        - Loads output projection after attention
        - Load FFN layer weights and biases
        - Load layer norm weights and biases
    3. Load final layer norm weights and biases
    4. Assign all loaded weights to the GPT model
"""

# gpt: instance of GPTModel
# params: dictionary of pretrained weights from OpenAI

def load_weights_into_gpt(gpt,params):

    # wpe: position embeddings
    # wte: token embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # Loop through each transformer block
    for b in range(len(params["blocks"])):

        # Split QKV weights and biases
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"]["w"]), 3, axis=-1
        )

        # Load Q projection weights
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        
        # Load K projection weights
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        
        # Load V projection weights
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        

        # Split the biases for Q,K,V
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"]["b"]), 3, axis=-1
        )

        # Load Q projection biases
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        
        # Load K projection biases
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        
        # Load V projection biases
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        

        # Load output projection weights
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )

        # Load output projection biases
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # Load FFN layer 1 weights and biases
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )

        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )

        # Load FFN layer 2 weights and biases
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )   
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # Load Layer Norm 1 weights and biases
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )

        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )

        # Load Layer Norm 2 weights and biases
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
    # Load final Layer Norm weights and biases
    gpt.final_norm.scale = assign(
        gpt.final_norm.scale,
        params["g"]
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift,
        params["b"]
    )
    gpt.out_head.weight = assign(
        gpt.out_head.weight,
        params["wte"]
    )
        
        