import torch as nn


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        input_dim: input noise dimensionality
        output_dim: output dimensionality
        n_layers: number of LSTM layers
        hidden_dim: dimensionality of the hidden layer of LSTM
    """

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        x = self.dense(x)
        x = self.lstm(x)
        return x


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects to output probabilities for each element.
    Args:
        input_dim: Input dimensionality
        n_layers: number of LSTM layers
        hidden_dim: dimensionality of the hidden layer of LSTM
    """

    def __init__(self, input_dim, n_layers):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
