import torch
from torch import nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)
        self.zero_input = torch.zeros(1, 16, self.hidden_size).to(device)

    def forward(self, question, hidden=None):
        embed = self.embed(question)
        if hidden is None:
            output, hidden = self.lstm(embed)
        else:
            output, hidden = self.lstm(embed, hidden)
        return output, hidden

    def train(self, question, answer, opt, crit):
        opt.zero_grad()
        loss = 0
        output, hidden = self.forward(question)
        output, hidden = self.lstm(self.zero_input, hidden)
        loss = crit(output[:, -1], answer[:, 0]-1)
        if answer.size(1) > 1:
            output, _ = self.forward(answer[:, :-1], hidden)
            loss += crit(output.permute(0, 2, 1), answer[:, 1:]-1)
        loss.backward()
        opt.step()
        return loss.item()

    # def forward(self, questions, questions_len, hiddens=None):
    #     embeds = self.embed(questions)
    #     embeds = nn.utils.rnn.pack_padded_sequence(embeds, questions_len, batch_first=True)
    #     if hiddens is None:
    #         outputs, hiddens = self.lstm(embeds)
    #     else:
    #         outputs, hiddens = self.lstm(embeds, hiddens)
    #     outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
    #     return outputs, hiddens

    def set_zero_state(self, b_size):
        self.zero_state = torch.zeros(b_size, 16, self.hidden_size).to(self.device)

    def train_batch(self, questions, questions_len, answers, answers_len,
        answer_mappings, opt, crit):
        b_size = questions.size(0)
        opt.zero_grad()
        _, (s1, s2) = self.forward(questions, questions_len, None)
        output, (s1, s2) = self.lstm(self.zero_state, (s1, s2))
        output = output.index_select(0, answer_mappings)
        s1 = s1.index_select(1, answer_mappings)
        s2 = s2.index_select(1, answer_mappings)
        loss = crit(output[:, -1, :], answers[:, 0]-1)
        output, _ = self.forward(answers, answers_len, (s1, s2))
        for i in range(output.size(0)):
            if answers_len[i] > 1:
                loss += crit(output[i, :answers_len[i]-1, :], answers[i, 1:answers_len[i]]-1)
        loss = loss / b_size
        loss.backward()
        opt.step()
        return loss.item()

# For debugging tensor operations.
if __name__ == '__main__':
    pass
    # b_size = 1024
    # q_len, a_len = 160, 60
    # vocab_size = 96
    # device = torch.device('cuda')
    # questions = torch.randint(0, vocab_size, (b_size, q_len), dtype=torch.int64).to(device)
    # questions_len = q_len * torch.ones((b_size), dtype=torch.long).to(device)
    # answers = torch.randint(0, vocab_size, (b_size, a_len), dtype=torch.int64).to(device)
    # answers_len = a_len * torch.ones((b_size), dtype=torch.long).to(device)
    # answer_mappings = torch.LongTensor(list(range(b_size))).to(device)
    # net = SimpleLSTM(device, vocab_size, vocab_size+1, 2048, vocab_size).to(device)
    # from torch import optim
    # opt = optim.Adam(net.parameters(), lr=1e-4)
    # crit = nn.CrossEntropyLoss()
    # net.set_zero_state(b_size)
    # net.train_batch(questions, questions_len, answers, answers_len, answer_mappings, opt, crit)
