import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

TRAIN_FILE = './data/train_2.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIME_STEP = 10;
INPUT_SIZE = 1;
HIDDEN_SIZE = 32;
DROPOUT_RATE = 0.0;

def read_data(train_file=TRAIN_FILE):
    df = pd.read_csv(train_file)
    print(df.head())


class EncoderRNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=DROPOUT_RATE,
        )

    def forward(self, input, h_state):
        output, h_n = self.gru(input, h_state)
        return output, h_n


class DecoderRNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, time_step=TIME_STEP):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.time_step = time_step

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=DROPOUT_RATE,
        )

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, input, h_state):
        h_output, h_n = self.gru(input, h_state)
        h_output_reshaped = h_output.view(-1, self.hidden_size)
        output = self.out(h_output_reshaped)
        output = output.view(-1, self.time_step, self.hidden_size)

        return output, h_n

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, time_step=TIME_STEP):
        super(Seq2SeqModel, self).__init__()

        self.encoder = EncoderRNN(input_size, hidden_size);
        self.decoder = DecoderRNN(input_size, hidden_size, time_step);

    def forward(self, en_input, de_input, h_state):
        # assume input (batch, time_step, input_size)
        en_out, en_h_n = self.encoder(en_input, h_state)
        de_out, de_h_n = self.decoder(de_input, en_h_n)

        return de_out, de_h_n


seq2seq_modal = Seq2SeqModel()
optimizer = torch.optim.Adam(seq2seq_modal.parameters(), lr=1e-2)
loss_func = nn.MSELoss()

def train(input_length, input_tensor, train_x, train_y):
    optimizer.zero_grad()

    for e in range(epoch):
        (x, y) = sample_batch(train_x, train_y)
        # x (batch, en_time_step, input_size)
        # y (batch, de_time_step, output_size)
        de_x = np.concatenate(x[-1:],y[:-1])
        h_0 = torch.zeros(1, x.shape[0], HIDDEN_SIZE, device=device)

        x = torch.from_numpy(x)
        de_x = torch.form_numpy(de_x)
        y = torch.from_numpy(y)

        autoencoder = Seq2SeqModal()
        de_out, de_h_n = Seq2SeqModal(x, de_x, h_0)

        loss = loss_func(de_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    read_data()


if __name__ == "__main__":
    main()
