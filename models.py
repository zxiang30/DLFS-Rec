import torch
import torch.nn as nn
from modules import LayerNorm, Encoder, VanillaAttention


class DLFSRecModel(nn.Module):
    def __init__(self, args):
        super(DLFSRecModel, self).__init__()
        self.args = args
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.side_mean_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)
        self.side_cov_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)

        if args.fusion_type == 'concat':
            self.mean_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)

        elif args.fusion_type == 'gate':
            self.mean_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)

        self.mean_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.cov_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.elu = torch.nn.ELU()

        self.apply(self.init_weights)

    def forward(self, input_ids, input_context):

        mean_id_emb = self.item_mean_embeddings(input_ids)
        cov_id_emb = self.item_cov_embeddings(input_ids)

        input_attrs = self.args.items_feature[input_ids]
        mean_side_dense = self.side_mean_dense(torch.cat((input_context, input_attrs), dim=2))
        cov_side_dense = self.side_cov_dense(torch.cat((input_context, input_attrs), dim=2))

        if self.args.fusion_type == 'concat':
            mean_sequence_emb = self.mean_fusion_layer(torch.cat((mean_id_emb, mean_side_dense), dim=2))
            cov_sequence_emb = self.cov_fusion_layer(torch.cat((cov_id_emb, cov_side_dense), dim=2))
        elif self.args.fusion_type == 'gate':
            mean_concat = torch.cat(
                [mean_id_emb.unsqueeze(-2), mean_side_dense.unsqueeze(-2)], dim=-2)
            mean_sequence_emb, _ = self.mean_fusion_layer(mean_concat)
            cov_concat = torch.cat(
                [cov_id_emb.unsqueeze(-2), cov_side_dense.unsqueeze(-2)], dim=-2)
            cov_sequence_emb, _ = self.cov_fusion_layer(cov_concat)
        else:
            mean_sequence_emb = mean_id_emb + mean_side_dense
            cov_sequence_emb = cov_id_emb + cov_side_dense

        mask = (input_ids > 0).long().unsqueeze(-1).expand_as(mean_sequence_emb)
        mean_sequence_emb = mean_sequence_emb * mask
        cov_sequence_emb = cov_sequence_emb * mask

        mean_sequence_emb = self.dropout(self.mean_layer_norm(mean_sequence_emb))
        cov_sequence_emb = self.elu(self.dropout(self.cov_layer_norm(cov_sequence_emb))) + 1

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                output_all_encoded_layers=True)
        sequence_mean_output, sequence_cov_output = item_encoded_layers[-1]

        return sequence_mean_output, sequence_cov_output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
