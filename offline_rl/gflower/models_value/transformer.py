import torch
import torch.nn as nn

### Generated code, not tested
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.register_buffer('positional_encoding', self._generate_positional_encoding(model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, output_dim)

    def _generate_positional_encoding(self, model_dim, max_len=1000):
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (B, T, C)
        
        Returns:
            Tensor: Output tensor of shape (B, T, C)
        """
        src = src.transpose(0, 1) # from (B, T, C) to (T, B, C)
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        src = src + self.positional_encoding[:src.size(0), :]
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output.transpose(0, 1) # from (T, B, C) to (B, T, C)
