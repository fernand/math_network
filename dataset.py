import glob
import os
from multiprocessing import Pool
import string

import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, ConcatDataset

DIRS = {
    # 'train': ['train-easy', 'train-medium', 'train-hard'],
    'train': ['train-easy'],
    'test': ['interpolate']
}

MAX_Q_LEN = 160
MAX_A_LEN = 60

# '\n' is the special end of sentence character
CHARS = string.ascii_letters + string.digits + string.punctuation + ' ' + '\n'
PADDING_IDX = 0
EOS_IDX = len(CHARS)
# Start index at 1. 0 is the padding index.
CHAR_DICT = dict(zip(CHARS, range(1, len(CHARS)+1)))
VOCAB_SIZE = len(CHAR_DICT)

def get_modules(root):
    module_files = glob.glob(os.path.join(root, DIRS['test'][0], '*.txt'))
    return sorted([f.split('/')[-1].split('.')[0] for f in module_files])

def gen_datasets(root, split):
    modules = get_modules(root)
    with Pool(6) as pool:
        r = list(tqdm(pool.imap(gen_dataset, [(root, split, m) for m in modules]), total=len(modules)))

def gen_dataset(args):
    root, split, module = args
    output = h5py.File(os.path.join(root, f'{split}_{module}.hdf5'), 'w')
    data = []
    for data_dir in DIRS[split]:
        with open(os.path.join(root, data_dir, module+'.txt')) as f:
            i = 0
            question = None
            for l in f:
                if i % 2 == 0:
                    question = l.strip()
                else:
                    answer = l.strip()
                    q_arr = np.zeros(MAX_Q_LEN, dtype=np.uint8)
                    q_arr[:len(question)] = np.array([CHAR_DICT[c] for c in question])
                    a_arr = np.zeros(MAX_A_LEN, dtype=np.uint8)
                    a_arr[:len(answer)] = np.array([CHAR_DICT[c] for c in answer])
                    data.append((q_arr, a_arr))
                i += 1
    questions = output.create_dataset('questions', (len(data), MAX_Q_LEN), dtype=np.uint8)
    answers = output.create_dataset('answers', (len(data), MAX_A_LEN), dtype=np.uint8)
    for i, x in enumerate(data):
        questions[i] = x[0]
        answers[i] = x[1]
    output.close()

def get_datasets(root, split):
    modules = get_modules(root)
    return ConcatDataset([ModuleDataset(root, split, m) for m in modules])

class ModuleDataset(Dataset):
    def __init__(self, root, split, module):
        self.path = os.path.join(root, f'{split}_{module}.hdf5')
        self.file = None
        with h5py.File(self.path, 'r') as f:
            self.len = len(f['questions'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        padded_q = self.file['questions'][index]
        padded_a = self.file['answers'][index]
        # Copy the data to change the dtype since nn.Embedding requires LongTensor.
        return (torch.LongTensor(padded_q[np.where(padded_q != 0)]),
            torch.LongTensor(padded_a[np.where(padded_a != 0)]))

    def __len__(self):
        return self.len

    def close(self):
        if self.file is not None:
            self.file.close()

def collate_data(batch):
    # Need to sort the questions by decreasing length so that they can be packed later.
    # Answers will have to be packed too.
    b_size = len(batch)
    sorted_batch = sorted([x for x in batch],
        key=lambda x: x[0].size(0), reverse=True)
    max_q_len = sorted_batch[0][0].size(0)
    questions = torch.zeros((b_size, max_q_len+1), dtype=torch.long)
    sorted_answers = sorted([(i, x[1]) for i, x in enumerate(sorted_batch)],
        key=lambda t: t[1].size(0), reverse=True)
    max_a_len = sorted_answers[0][1].size(0)
    answers = torch.zeros((b_size, max_a_len+1), dtype=torch.long)
    questions_len = torch.zeros((b_size), dtype=torch.long)
    answers_len = torch.zeros((b_size), dtype=torch.long)
    answer_mappings = torch.zeros((b_size), dtype=torch.long)
    for i, x in enumerate(sorted_batch):
        q_len = x[0].size(0)
        questions_len[i] = q_len
        questions[i, :q_len] = x[0]
        questions[i, q_len] = EOS_IDX
    for i, t in enumerate(sorted_answers):
        idx, a = t
        a_len = a.size(0)
        answers_len[i] = a_len
        answers[i, :a_len] = a
        answers[i, a_len] = EOS_IDX
        answer_mappings[idx] = i
    return questions, questions_len, answers, answers_len, answer_mappings

if __name__ == '__main__':
    PATH = '/home/fernand/math/data'
    gen_datasets(PATH, 'train')
    # gen_datasets(PATH, 'test')
