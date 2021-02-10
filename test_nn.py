##############################################################################
##############################################################################
########                    SHRI PROJECT                       ###############
########                    Prof. Nardi                        ###############
########                                                       ###############
########            Student: FRANCESCO CASSINI                 ###############
########            Sapienza ID: 785771                        ###############
########     Master in Roboics and Artificial Intelligence     ###############
##############################################################################
##############################################################################
########    
##############################################################################
##############################################################################


#https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb
import torch
import torchtext
from torchtext import data
from torchtext import datasets

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import spacy
from torchtext.data.utils import get_tokenizer
# tokenizer = get_tokenizer("spacy")


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"))
LABEL = data.LabelField()
LANGUAGE = data.LabelField()
fields = {
  'text': ('text', TEXT),
  'label': ('label', LABEL),
  'language': ('language', LANGUAGE),
}

train_data, test_data = data.TabularDataset.splits(
  path = '',
  train = 'trainset.json',
  validation = 'testset.json',
  format = 'json',
  fields = fields,
)

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.300d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)
LANGUAGE.build_vocab(train_data)
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Intent_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [sent len, batch size]
        text = text.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = Intent_model(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
model.load_state_dict(torch.load('tut5-model.pt'))
nlp = spacy.load("en_core_web_sm")

def predict_class(model, sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()


phrase = "Recover data from drive please"
pred_class = predict_class(model, phrase)
print('Phrase to predict: '+ '"'+ phrase+ '"')
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

phrase = "Mantain my doc thanks"
pred_class = predict_class(model, phrase)
print('Phrase to predict: '+ '"'+ phrase+ '"')
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

phrase = "Load this document now"
pred_class = predict_class(model, phrase)
print('Phrase to predict: '+ '"'+ phrase+ '"')
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

phrase = "preserve my file in archive"
pred_class = predict_class(model, phrase)
print('Phrase to predict: '+ '"'+ phrase+ '"')
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

