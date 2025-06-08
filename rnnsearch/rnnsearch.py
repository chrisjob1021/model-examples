"""
RNNsearch: Neural Machine Translation by Jointly Learning to Align and Translate
Implementation based on the paper: https://arxiv.org/pdf/1409.0473.pdf

This is a minimal implementation of the attention-based neural machine translation
model proposed by Bahdanau et al. (2014). The model uses a bidirectional RNN
encoder and a decoder with an attention mechanism to jointly learn translation
and alignment between source and target sequences.

Key features:
- Bidirectional RNN encoder (GRU)
- Attention-based decoder with GRU
- Joint learning of translation and alignment
- Maxout output layer
"""

import torch
from torch import nn

# Set device to GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        # Initialize embedding layer to convert token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Initialize bidirectional GRU for sequence encoding
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, T_x) - batch of input sequences where T_x is the length of the source sequence
        # Convert token IDs to embeddings
        x = self.embedding(x)
        # Process through GRU, get hidden states for each time step
        h_j, _ = self.gru(x)
        return h_j  # (B, T_x, 2*hidden_dim) - bidirectional states

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        # Initialize embedding layer for target tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # GRU takes concatenated embedding and context vector
        self.gru = nn.GRU(embed_dim + 2 * hidden_dim, hidden_dim, batch_first=True)
        # Attention mechanism components
        self.W_a = nn.Linear(hidden_dim + 2 * hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)
        # Maxout output layer (2 pieces)
        self.maxout = nn.Linear(hidden_dim + 2 * hidden_dim, 2 * vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, y, h_j):
        # y: (B, T_y) - target sequence including start token
        B, T_x, _ = h_j.size()
        # Initialize decoder hidden state and context vector
        h_i = torch.zeros(1, B, self.hidden_dim, device=h_j.device)  # s_{i-1} - first dimension is num_layers=1 for GRU
        c_i = torch.zeros(B, 2 * self.hidden_dim, device=h_j.device)

        outputs = []
        for t in range(y.size(1)):
            # "At each time step i, the input to the decoder is the embedding of the previous target symbol y_{i-1}."
            y_prev = y[:, t]
            y_emb = self.embedding(y_prev).unsqueeze(1)  # (B, 1, embed_dim) - second dimension is sequence length

            # The input to the decoder at each time step is the concatenation of the embedding of the previous target symbol and the context vector c_i
            gru_in = torch.cat([y_emb, c_i.unsqueeze(1)], dim=2)  # (B, 1, embed_dim + 2*hidden_dim)

            # s_i is the decoder's hidden state at time step i, representing the current state of the decoder
            s_i, h_i = self.gru(gru_in, h_i)  
            s_i = s_i.squeeze(1)  # (B, hidden_dim) - Remove the batch dimension since we're processing one token at a time

            # Compute attention scores
            # Expand decoder state s_i to match encoder states h_j dimensions for attention computation
            # s_i: (B, hidden_dim) -> s_i_rep: (B, T_x, hidden_dim)
            # We repeat s_i T_x times because we need to compute attention scores between the current decoder state s_i and each encoder state h_j
            # This allows you to compute the attention "energy" for every encoder position for the current decoder step, regardless of length of the target vs source sequence
            s_i_rep = s_i.unsqueeze(1).repeat(1, T_x, 1)

            # Compute energy scores for each encoder state
            # - s_i_rep: (B, T_x, hidden_dim) - repeated decoder state
            # - h_j: (B, T_x, 2*hidden_dim) - encoder states
            # - W_a: (2*hidden_dim, hidden_dim) - weight matrix for attention
            # - tanh(W_a(s_i_rep, h_j)): (B, T_x, hidden_dim) - transformed energy scores
            # - v_a: (hidden_dim, 1) - bias vector for attention
            # We squeeze the last dimension (-1) because v_a produces a tensor of shape (B, T_x, 1)
            # and we want to remove this singleton dimension to get (B, T_x) for the energy scores
            energy = self.v_a(torch.tanh(self.W_a(torch.cat([s_i_rep, h_j], dim=2)))).squeeze(-1)

            # Get attention weights using softmax to normalize scores to probabilities  
            # alpha_ij: (B, T_x) - attention weights for each encoder state
            alpha_ij = torch.softmax(energy, dim=1)
            
            # Compute context vector as weighted sum of encoder states
            # - alpha_ij: (B, 1, T_x) - attention weights
            # - h_j: (B, T_x, 2*hidden_dim) - encoder states
            # When you multiply each batch: (1, T_x) x (T_x, 2hidden_dim) → (1, 2hidden_dim)
            # - c_i: (B, 2*hidden_dim) - context vector

            # bmm is a batch matrix multiplication, which is a more efficient way to compute the context vector than a for loop.
            c_i = torch.bmm(alpha_ij.unsqueeze(1), h_j).squeeze(1)  # (B, 2*hidden_dim)

            # Maxout is a non-linear activation function that takes the maximum of multiple linear transformations
            # Here we concatenate decoder state and context vector
            #  s_i: (B, hidden_dim), c_i: (B, 2*hidden_dim)
            # Concatenation result: (B, 3*hidden_dim)
            o_i = self.maxout(torch.cat([s_i, c_i], dim=1)) # (B, 2 * vocab_size)

            # Reshape for maxout operation
            # - After max: (B, 2 * vocab_size) - take maximum of each group
            # - After view: (B, 2, vocab_size) - groups of 2 features
            # .max(dim=1)[0] takes the maximum of the two for each vocabulary entry
            o_i = o_i.view(B, 2, -1).max(dim=1)[0]
            # o_i shape: (B, vocab_size) - batch of output logits for each vocabulary token
            outputs.append(o_i)
        # Stack outputs along sequence dimension (dim=1) to create tensor of shape (B, T_y, vocab_size)
        return torch.stack(outputs, dim=1)

class RNNsearch(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=16, hidden_dim=32):
        super().__init__()
        # Initialize encoder and decoder components
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim)

    def forward(self, x, y):
        # Encode source sequence
        h_j = self.encoder(x)
        # Decode target sequence with attention
        logits = self.decoder(y, h_j)
        # logits shape: (B, T_y, vocab_size) where:
        # - B: batch size
        # - T_y: target sequence length
        # - vocab_size: size of target vocabulary
        return logits

# ---------------------------
# Toy dataset (reverse task)
# ---------------------------

def generate_batch(batch_size=16, max_len=5, vocab_size=12, start_token=10):
    # Generate random sequence lengths for each example
    lengths = torch.randint(3, max_len + 1, (batch_size,))
    # Initialize input and target tensors
    x = torch.zeros(batch_size, max_len, dtype=torch.long)
    y = torch.zeros(batch_size, max_len + 1, dtype=torch.long)
    # Loop through each sequence in the batch
    for i, L in enumerate(lengths):
        # Generate random tokens for input sequence, excluding padding (0) and start token
        tokens = torch.randint(1, vocab_size - 1, (L,))
        # Fill input sequence up to its length L
        x[i, :L] = tokens
        # Set start token at beginning of target sequence
        y[i, 0] = start_token
        # Fill target sequence with reversed input tokens, starting after start token
        # It's a toy example, which is why we're using a reversed sequence
        y[i, 1:L + 1] = tokens.flip(0)
    return x.to(DEVICE), y.to(DEVICE)

if __name__ == "__main__":
    # Set vocabulary sizes
    SRC_VOCAB = 12
    TGT_VOCAB = 12
    # Initialize model and move to device
    model = RNNsearch(SRC_VOCAB, TGT_VOCAB).to(DEVICE)
    # Set up optimizer and loss function
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for step in range(50):
        # Generate batch of examples
        src, tgt = generate_batch()
        # Clear previous gradients
        optim.zero_grad()
        # Forward pass (excluding last target token)
        # The last token in the target sequence has no "next" token to predict, so it is not used as an input during training.
        logits = model(src, tgt[:, :-1]) # (B, T_y, vocab_size)
        
        # Compute loss (excluding first target token which is start token)
        # Reshape logits from (B, T_y, vocab_size) to (B*T_y, vocab_size) for cross entropy
        # Reshape targets from (B, T_y) to (B*T_y) to match logits
        loss = loss_fn(logits.reshape(-1, TGT_VOCAB), tgt[:, 1:].reshape(-1))
        # Backward pass and optimization
        loss.backward()
        optim.step()
        if (step + 1) % 10 == 0:
            print(f"step {step+1:3d} | loss {loss.item():.3f}")
