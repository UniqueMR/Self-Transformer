import pickle
import pandas as pd
import torch
from torchtext.legacy import data
import re
import spacy
import os
import numpy as np
from torch.autograd import Variable

def read_data():
    src_data = './data/english.txt'
    trg_data = './data/french.txt'

    src_data = open(src_data, encoding='utf-8').read().strip().split('\n')
    trg_data = open(trg_data, encoding='utf-8').read().strip().split('\n')

    return src_data, trg_data

class tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

def create_fields(load_weights):
    print("loading spacy tokenizers...")
    t_src = tokenize('en_core_web_sm')
    t_trg = tokenize('fr_core_news_sm')

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='eos')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if load_weights is not None:
        print("loading presaved fields...")
        SRC = pickle.load(open(f'./weights/SRC.pkl', 'rb'))
        TRG = pickle.load(open(f'./weights/TRG.pkl', 'rb'))

    return (SRC, TRG)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

def create_dataset(SRC, TRG, src_data, trg_data):
    print("creating dataset and iterator...")

    raw_data = {'src' : [line for line in src_data], 'trg' : [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])# df means dataframe

    max_strlen = 80
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=1500, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True, shuffle=True)
    os.remove('translate_transformer_temp.csv')

    SRC.build_vocab(train)
    TRG.build_vocab(train)

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    train_len = get_len(train_iter)

    return train_iter, train_len, src_pad, trg_pad

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, src_pad, trg_pad):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask
