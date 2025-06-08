import torch
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, T_x)
        x = self.embedding(x)
        h_j, _ = self.gru(x)
        return h_j  # (B, T_x, 2*hidden_dim)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + 2 * hidden_dim, hidden_dim, batch_first=True)
        self.W_a = nn.Linear(hidden_dim + 2 * hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)
        self.out = nn.Linear(hidden_dim + 2 * hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, y, h_j):
        # y: (B, T_y) - target sequence including <s>
        B, T_x, _ = h_j.size()
        h_i = torch.zeros(1, B, self.hidden_dim, device=h_j.device)  # s_{i-1}
        c_i = torch.zeros(B, 2 * self.hidden_dim, device=h_j.device)

        outputs = []
        for t in range(y.size(1)):
            y_prev = y[:, t]
            y_emb = self.embedding(y_prev).unsqueeze(1)  # (B, 1, embed_dim)
            gru_in = torch.cat([y_emb, c_i.unsqueeze(1)], dim=2)
            s_i, h_i = self.gru(gru_in, h_i)  # s_i: (B, 1, hidden_dim)
            s_i = s_i.squeeze(1)

            # alignment model e_{ij}
            s_i_rep = s_i.unsqueeze(1).repeat(1, T_x, 1)
            energy = self.v_a(torch.tanh(self.W_a(torch.cat([s_i_rep, h_j], dim=2)))).squeeze(-1)
            alpha_ij = torch.softmax(energy, dim=1)  # (B, T_x)
            c_i = torch.bmm(alpha_ij.unsqueeze(1), h_j).squeeze(1)  # (B, 2*hidden_dim)

            o_i = self.out(torch.cat([s_i, c_i], dim=1))
            outputs.append(o_i)
        return torch.stack(outputs, dim=1)

class RNNsearch(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim)

    def forward(self, x, y):
        h_j = self.encoder(x)
        logits = self.decoder(y, h_j)
        return logits

# ---------------------------
# Toy dataset (reverse task)
# ---------------------------

def generate_batch(batch_size=16, max_len=5, vocab_size=12, start_token=10):
    lengths = torch.randint(3, max_len + 1, (batch_size,))
    x = torch.zeros(batch_size, max_len, dtype=torch.long)
    y = torch.zeros(batch_size, max_len + 1, dtype=torch.long)
    for i, L in enumerate(lengths):
        tokens = torch.randint(1, vocab_size - 1, (L,))
        x[i, :L] = tokens
        y[i, 0] = start_token
        y[i, 1:L + 1] = tokens.flip(0)
    return x.to(DEVICE), y.to(DEVICE)

if __name__ == "__main__":
    SRC_VOCAB = 12
    TGT_VOCAB = 12
    model = RNNsearch(SRC_VOCAB, TGT_VOCAB).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for step in range(50):
        src, tgt = generate_batch()
        optim.zero_grad()
        logits = model(src, tgt[:, :-1])
        loss = loss_fn(logits.reshape(-1, TGT_VOCAB), tgt[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        if (step + 1) % 10 == 0:
            print(f"step {step+1:3d} | loss {loss.item():.3f}")
