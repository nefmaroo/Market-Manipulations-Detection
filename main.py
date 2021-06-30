import random
import numpy as np
import pandas as pd
import torch as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import LSTMGenerator, LSTMDiscriminator


def train(train, generator, discriminator, n_epochs, batch_size):
    sequence_len = train.size(2)
    for epoch in range(n_epochs):
        order = np.random.permutation(len(train))
        for start_index in range(0, len(train), batch_size):
            # Sample from generator
            noise = torch.randn(batch_size * sequence_len)
            fake = generator(noise)

            batch_indexes = order[start_index:start_index + batch_size]
            batch = train[batch_indexes].to(device)

            # Train the discriminator
            fake_dscore = discriminator(fake)
            true_dscore = discriminator(batch)

            floss = criterionD(fake_dscore, torch.zeros_like(fake_dscore))
            tloss = criterionD(true_dscore, torch.ones_like(true_dscore))
            dloss = floss + tloss

            dloss.backward()
            optimizerD.step()
            optimizerD.zero_grad()

            # Sample again from the generator and get output from discriminator
            noise = torch.randn(batch_size * sequence_len)
            fake = generator(noise)
            fake_dscore = discriminator(fake)

            # Train the generator
            gloss = criterionG(fake_dscore, torch.zeros_like(fake_dscore))
            gloss.backward()
            optimizerG.step()
            optimizerG.zero_grad()


def test(test, n_epoch, batch_size, discriminator):
    classification_error = []
    f_score = []
    for epoch in range(n_epochs):
        class_err = []
        f = []
        order = np.random.permutation(len(test))
        for start_index in range(0, len(test), batch_size):
            batch_indexes = order[start_index:start_index + batch_size]
            batch = test[batch_indexes].to(device)

            batch_res = discriminator(batch)
            class_err.append(accuracy_score(batch_res, torch.zeros_like(batch_res)))
            f.append(precision_recall_fscore_support(batch_res, torch.zeros_like(batch_res)))
        classification_error.append(np.mean(class_err))
        f_score.append(np.mean(f))
    return classification_error, f_score


random.seed(0)
np.random.seed(0)
nn.manual_seed(0)
nn.cuda.manual_seed(0)
nn.backends.cudnn.deterministic = True

df = pd.read_csv('data.csv')
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
train_data = nn.reshape(nn.tensor(train_data.to_numpy(), (4, -1, 15))
test_data.to_csv('test_data.csv')

device = nn.device('cuda:0' if nn.cuda.is_available() else 'cpu')

generator = LSTMGenerator(input_dim=200, out_dim=1, hidden_dim=256).to(device)
discriminator = LSTMDiscriminator(input_dim=1, hidden_dim=256).to(device)

criterionD = nn.BCELoss()
optimizerD = nn.optim.Adam(generator.parameters())

criterionG = torch.nn.MSE()
optimizerG = nn.optim.Adam(discriminator.parameters())

train(train_data, 50, 100)

# Test unseen normal cases
test_data = nn.reshape(nn.tensor(test_data.to_numpy(), (4, -1, 15))
test(test_data, 50, 100, discriminator)

# Test unseen normal cases with manipulative patterns injection
test_data = pd.read_csv('manipulative_data.csv')
test_data = nn.reshape(nn.tensor(test_data.to_numpy(), (4, -1, 15))
test(test_data, 50, 100, discriminator)