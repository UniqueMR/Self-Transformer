from torchtext.legacy import data
import re
import spacy

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

def create_fields():
    print("loading spacy tokenizers...")
    t_src = tokenize('en_core_web_sm')
    t_trg = tokenize('fr_core_news_sm')

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='eos')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    return (SRC, TRG)

SRC, TRG = create_fields()
print(SRC)
print(TRG)

