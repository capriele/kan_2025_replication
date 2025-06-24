import torch
import torch.nn as nn

input_size = 5000
hidden_layers = 20
output_size = 5000


class LSTMAnomaly(nn.Module):
    def __init__(
        self, input_dim=input_size, hidden_dim=hidden_layers, output_dim=output_size
    ):
        super(LSTMAnomaly, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# === Autoencoder Model ===
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, output_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
