import torch
import torch.nn as nn
import copy


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_num=256,
        feature_dim=256,
        decoder_num=12,
        decode_layer_num=12,
        n_heads=8,
    ):
        super(GPTDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_num = max_seq_num
        self.feature_dim = feature_dim
        self.decoder_num = decoder_num
        self.decode_layer_num = decode_layer_num
        self.n_heads = n_heads

        self.token_embedding = nn.Embedding(self.vocab_size, self.feature_dim)
        self.position_embedding = nn.Embedding(self.max_seq_num, self.feature_dim)
        # decode layer just use masked attention, so it can be replaced with encoder layer of pytorch native
        self.decoder_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=self.n_heads
            ),
            num_layers=self.decode_layer_num,
        )
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        self.token_ffn = nn.Linear(self.feature_dim, self.vocab_size)

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tokens):
        token_feature = self.token_embedding(tokens)
        position_feature = self.position_embedding(
            torch.arange(self.max_seq_numa)
        ).type_as(token_feature)
        input_feature = token_feature + position_feature

        src_mask = self.generate_square_subsequent_mask(self.max_seq_num)

        input_feature = input_feature.transpose(1,0) # b s d -> s b d

        #decoder weight sharing
        for _ in range(self.decoder_num):
            input_feature = self.decoder_block(input_feature, src_mask)
            input_feature = self.layer_norm(input_feature)

        input_feature = input_feature.transpose(0, 1) # s b d -> b s d

        token_preds = self.token_ffn(input_feature)
        return token_preds
        
