import copy
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class DistFilterLayer(nn.Module):
    def __init__(self, args):
        super(DistFilterLayer, self).__init__()
        self.mean_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.cov_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_mean_tensor, input_cov_tensor):
        batch, seq_len, hidden = input_mean_tensor.shape

        mean_x = torch.fft.rfft(input_mean_tensor, dim=1, norm='ortho')
        mean_weight = torch.view_as_complex(self.mean_complex_weight)
        mean_x = mean_x * mean_weight
        mean_sequence_emb_fft = torch.fft.irfft(mean_x, n=seq_len, dim=1, norm='ortho')
        mean_hidden_states = self.out_dropout(mean_sequence_emb_fft)
        mean_hidden_states = self.layer_norm(mean_hidden_states + input_mean_tensor)

        cov_x = torch.fft.rfft(input_cov_tensor, dim=1, norm='ortho')
        cov_weight = torch.view_as_complex(self.cov_complex_weight)
        cov_x = cov_x * cov_weight
        cov_sequence_emb_fft = torch.fft.irfft(cov_x, n=seq_len, dim=1, norm='ortho')
        cov_hidden_states = self.out_dropout(cov_sequence_emb_fft)
        cov_hidden_states = self.layer_norm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filter_layer = DistFilterLayer(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states):
        mean_filter_output, cov_filter_output = self.filter_layer(mean_hidden_states, cov_hidden_states)
        return mean_filter_output, self.activation_func(cov_filter_output) + 1


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            mean_hidden_states, cov_hidden_states = layer_module(mean_hidden_states, cov_hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        return all_encoder_layers


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights
