import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMStepwiseFusion(nn.Module):
    def __init__(
        self,
        init_signal_size=64,  # signal for initial states
        driven_signal_size=200,  # step-wise driven signal
        lstm_input_size=192,
        hidden_size=96,
        num_layers=4,
        bias=False,
        batch_first=True,
        dropout=0.1,
        bidirectional=False,
        output_size=64
    ):
        super().__init__()

        if bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.initial_fc = nn.Sequential(
            nn.LayerNorm(init_signal_size),
            nn.Linear(init_signal_size, init_signal_size),
            nn.Tanh(),
            nn.Linear(init_signal_size, 2 * self.D * num_layers * hidden_size))  # 2 for hidden and cell

        self.fusion_fc = nn.Sequential(
            nn.LayerNorm(driven_signal_size),
            nn.Linear(driven_signal_size, lstm_input_size),
            nn.Tanh())

        self.lstm_layers = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional)  # D = 1

        self.dynam_3dmm_fc = nn.Linear(self.D * hidden_size, output_size)

        self.lstm_num_layers = num_layers
        self.lstm_hidden_size = hidden_size

    def forward(
        self,
        init,  # input init: 1st frame 3DMM coefficients, size=(N, 1, H_coeff)
        driven,  # (N, L, C_driven) step-wise driven signal (audio only / audio+listener)
        lengths=None,  # # type: list(int)
    ):
        # input init signal to init_hidden and init_cell of size (D*num_layers, N, hidden_size).
        init_ = self.initial_fc(init)
        init_hidden, init_cell = init_.view(-1, 2, self.D * self.lstm_num_layers, self.lstm_hidden_size).permute(1, 2, 0, 3).contiguous()
        # driven signal to lstm_input
        fused_features = self.fusion_fc(driven)  # (N, L, lstm_input_size)
        # predict through LSTM layers
        if lengths is not None:  # (rnn_utils for computational efficiency)
            fused_features = rnn_utils.pack_padded_sequence(fused_features, lengths, batch_first=True, enforce_sorted=False)
            fused_features, _ = self.lstm_layers(fused_features, (init_hidden, init_cell))
            fused_features_unpacked, length_unpacked = rnn_utils.pad_packed_sequence(fused_features, batch_first=True)
        else:
            fused_features_unpacked, _ = self.lstm_layers(fused_features, (init_hidden, init_cell))

        pred = self.dynam_3dmm_fc(fused_features_unpacked)  # size=(N, L, output_size)
        return pred

