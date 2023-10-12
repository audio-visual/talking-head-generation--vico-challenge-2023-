import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math


class Transformer3DMM(nn.Module):
    def __init__(self,
                 embed_len=90,
                 audio_size=[40, 5],
                 d_init=64,
                 nhead=4,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 batch_first=True,
                 upper_face3d_indices=tuple(range(14, 64)),
                 lower_face3d_indices=tuple(range(14)),
                 ):
        super(Transformer3DMM, self).__init__()
        print("in Transformer3DMM init")

        self.upper_face3d_indices = upper_face3d_indices
        self.lower_face3d_indices = lower_face3d_indices
        self.C_exp = len(self.upper_face3d_indices) + len(self.lower_face3d_indices)
        assert self.C_exp == 64, "Expression dim should be 64."

        d_model = audio_size[0] * audio_size[1] + d_init

        self.init_encoding = InitConditionalEncoding(d_init=d_init, dropout=0.1, embed_len=embed_len)

        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=embed_len)

        self.transformer_upper = nn.Transformer(d_model=d_model,
                                                nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                batch_first=batch_first)

        self.transformer_lower = nn.Transformer(d_model=d_model,
                                                nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                batch_first=batch_first)

        tail_hidden_dim = d_model // 2
        self.upper_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(upper_face3d_indices)),
        )
        self.lower_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(lower_face3d_indices)),
        )

    def forward(self, content, init=None):
        """
        Args:
            content (_type_): driven_audio, (B, L_clip, n_mels, window)
            exp_init (_type_): (B, C_exp)

        Returns:
            face3d: (B, L_clip, C_exp)
        """

        B, T, C, W = content.shape  # B, T, 40, 5
        # print("content",content.shape)
        content = content.reshape(B, T, -1).permute(1, 0, 2)  # (T, B, C*W)

        # concatenate init condition
        init = init.unsqueeze(dim=0).expand(T, B, -1)  # (T, B, C_exp)
        init = self.init_encoding(init)

        content = torch.cat([content, init], dim=-1)  # (T, B, d_model)

        pos_embed = self.pos_encoding(content).permute(1, 0, 2)  # (B, T, d_model)

        upper_face3d_feat = self.transformer_upper(src=pos_embed, tgt=pos_embed)  # (B, T, d_model)
        upper_face3d = self.upper_tail_fc(upper_face3d_feat)  # (B, T, C_upper)

        lower_face3d_feat = self.transformer_lower(src=pos_embed, tgt=pos_embed)
        lower_face3d = self.lower_tail_fc(lower_face3d_feat)

        face3d = torch.zeros(B, T, self.C_exp).to(upper_face3d)
        face3d[:, :, self.upper_face3d_indices] = upper_face3d
        face3d[:, :, self.lower_face3d_indices] = lower_face3d

        return face3d


class InitConditionalEncoding(nn.Module):
    def __init__(self, d_init=64, dropout=0.1, embed_len=90):
        super(InitConditionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, embed_len, dtype=torch.float)
        position = 1 - 0.5 * position / len(position)  # Linear schedule
        position = position.unsqueeze(dim=1).expand(-1, d_init)  # (embed_len, d_init)
        self.register_buffer('position', position)

    def forward(self, init):
        """
        Args: init (_type_): (T, B, d_init)
        """
        multiplier = self.position.unsqueeze(dim=1)  # (embed_len, 1, d_init)
        init = init * multiplier[:init.shape[0], :, :]  # (T, B, d_init)
        return self.dropout(init)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

