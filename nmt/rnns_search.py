import torch
from torch import nn


def print_shape(name, tensor):
    """Utility to print the shape of a tensor."""
    print(f"{name}: {tuple(tensor.shape)}")


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        # src: (B, src_len)
        embedded = self.embedding(src)
        print_shape("encoder embedded", embedded)
        outputs, hidden = self.gru(embedded)
        print_shape("encoder outputs", outputs)
        print_shape("encoder hidden", hidden)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, H)
        # encoder_outputs: (B, src_len, 2H)
        src_len = encoder_outputs.size(1)
        dec = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(dec))
        scores = self.v(energy).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        print_shape("attention weights", attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        print_shape("context vector", context)
        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim * 3 + emb_dim, vocab_size)

    def forward(self, trg, encoder_outputs, hidden):
        # trg: (B, trg_len)
        B, trg_len = trg.size()
        outputs = []
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            embedded = self.embedding(input_token).unsqueeze(1)
            print_shape(f"decoder embedded step {t}", embedded)
            context = self.attention(hidden.squeeze(0), encoder_outputs)
            rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
            print_shape(f"decoder rnn input step {t}", rnn_input)
            output, hidden = self.gru(rnn_input, hidden)
            print_shape(f"decoder rnn output step {t}", output)
            pred = self.out(torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1))
            print_shape(f"decoder logits step {t}", pred)
            outputs.append(pred.unsqueeze(1))
            input_token = trg[:, t]
        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 2
    src_len = 5
    trg_len = 6  # includes <sos> token at position 0
    vocab_size = 10
    emb_dim = 8
    hidden_dim = 16

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))
    print_shape("source", src)
    print_shape("target", trg)

    encoder = Encoder(vocab_size, emb_dim, hidden_dim)
    enc_outputs, enc_hidden = encoder(src)

    # Combine bidirectional hidden states
    dec_hidden = torch.tanh(enc_hidden[0] + enc_hidden[1]).unsqueeze(0)
    print_shape("decoder initial hidden", dec_hidden)

    decoder = Decoder(vocab_size, emb_dim, hidden_dim)
    outputs = decoder(trg, enc_outputs, dec_hidden)
    print_shape("decoder outputs", outputs)
