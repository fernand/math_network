import os
import shutil

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_datasets, collate_data, VOCAB_SIZE, ModuleDataset
from simple_lstm import SimpleLSTM

PATH = '/home/fernand/math/data'
DEVICE = torch.device('cuda')
torch.manual_seed(12345)
BATCH_SIZE = 1024

def train_batch(epoch, net, opt, crit, batch_size):
    train_set = get_datasets('/home/fernand/math/data', 'train')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,
        collate_fn=collate_data, drop_last=True
    )
    pbar = tqdm(iter(train_loader))
    moving_loss = 0.0
    for questions, questions_len, answers, answers_len, answer_mappings in pbar:
        questions, questions_len = questions.to(DEVICE), questions_len.to(DEVICE)
        answers, answers_len = answers.to(DEVICE), answers_len.to(DEVICE)
        answer_mappings = answer_mappings.to(DEVICE)
        loss = net.train_batch(questions, questions_len, answers, answers_len,
            answer_mappings, opt, crit)
        if moving_loss == 0.0:
            moving_loss = loss
        else:
            moving_loss = 0.95 * moving_loss + 0.05 * loss
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        pbar.set_description('Epoch: {}; Loss: {:.5f}'.format(epoch + 1, moving_loss))
    for d in train_set.datasets:
        d.close()

def train_one(epoch, net, opt, crit):
    train_set = get_datasets('/home/fernand/math/data', 'train')
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=6)
    pbar = tqdm(iter(train_loader))
    moving_loss = 0.0
    for question, answer in pbar:
        question, answer = question.to(DEVICE), answer.to(DEVICE)
        loss = net.train(question, answer, opt, crit)
        if moving_loss == 0.0:
            moving_loss = loss
        else:
            moving_loss = 0.9999 * moving_loss + 0.0001 * loss
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        pbar.set_description('Epoch: {}; Loss: {:.5f}'.format(epoch + 1, moving_loss))
    for d in train_set.datasets:
        d.close()

if __name__ == '__main__':
    if os.path.exists('checkpoint'):
        shutil.rmtree('checkpoint')
    os.mkdir('checkpoint')

    net = SimpleLSTM(DEVICE, VOCAB_SIZE+1, 2048, VOCAB_SIZE).to(DEVICE)
    opt = optim.SGD(net.parameters(), lr=0.01)
    # opt = optim.Adam(net.parameters(), lr=6e-4, betas=(0.9, 0.995), eps=1e-9)
    crit = nn.CrossEntropyLoss(reduction='sum')
    net.set_zero_state(BATCH_SIZE)
    for epoch in range(20):
        # train_batch(epoch, net, opt, crit, BATCH_SIZE)
        train_one(epoch, net, opt, crit)
        with open(
            'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
        ) as f:
            torch.save(net.state_dict(), f)
