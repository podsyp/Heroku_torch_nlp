import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"

class ModelParams():
    def __init__(self):
        self.kernel_sizes = [3, 5, 6]
        self.vocab_size = 202065
        self.out_channels=16
        self.dropout = 0.25
        self.dim = 50
        self.patience=3


class CNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim,
            out_channels,
            kernel_sizes,
            dropout=0.5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.out_channels = out_channels

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                kernel_size=(kernel_sizes[0], emb_dim), padding=1, stride=2)  # YOUR CODE GOES HERE
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                kernel_size=(kernel_sizes[1], emb_dim), padding=1, stride=2)  # YOUR CODE GOES HERE
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                kernel_size=(kernel_sizes[2], emb_dim), padding=1, stride=2)  # YOUR CODE GOES HERE

        self.fc = nn.Linear(len(kernel_sizes) * out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)

        batch_size = embedded.shape[0]
        embedded = embedded.unsqueeze(1)  # may be reshape here

        conved_0 = F.relu(self.conv_0(embedded)).view(batch_size, self.out_channels, -1)  # may be reshape here
        conved_1 = F.relu(self.conv_1(embedded)).view(batch_size, self.out_channels, -1)  # may be reshape here
        conved_2 = F.relu(self.conv_2(embedded)).view(batch_size, self.out_channels, -1)  # may be reshape here

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        return self.fc(cat)

def load_model(path='data/model.pt'):
    params = ModelParams()
    model = CNN(vocab_size=params.vocab_size, emb_dim=params.dim, out_channels=params.out_channels,
                kernel_sizes=params.kernel_sizes, dropout=params.dropout)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
