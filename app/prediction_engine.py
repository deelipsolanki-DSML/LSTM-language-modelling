import torch
import torch.nn as nn
import numpy as np
import json
from string import punctuation
punc = set(punctuation)
punc.discard("\'")

with open("artifacts/vocab.json", 'r') as file:
    vocab = json.load(file)
len(vocab)

def text_to_indices(sentence, vocab):
  numerical_sentence = []
  for token in sentence:
    if token in vocab:
      numerical_sentence.append(vocab[token])
    else:
      numerical_sentence.append(vocab['<unk>'])
  return numerical_sentence

class LSTMModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 100)
    self.lstm = nn.LSTM(input_size=100, hidden_size=200,batch_first=True)
    self.fc = nn.Linear(200, vocab_size)

  def forward(self, x):
    embedded = self.embedding(x)
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
    output = self.fc(final_hidden_state.squeeze(0))
    return output

#laod model
model = LSTMModel(vocab_size=len(vocab))

model.load_state_dict(
    torch.load("artifacts/cornell_lstm_next_word.pth", map_location="cpu", weights_only=True)
)
model.eval()

# prediction
def prediction(text, top_k):
  model.eval()

  # tokenize
  def process(text):
    #lowercasing, punc, digit, special character removal
    new_text = ""
    for i in text:
        if i not in punc and not i.isdigit():
            new_text+=i
    tokens = new_text.lower().split()
    return tokens

  tokenized_text = process(text)

  # text -> numerical indices
  numerical_text = text_to_indices(tokenized_text, vocab)
  padded_text = torch.tensor(numerical_text, dtype=torch.long).unsqueeze(0).to('cpu')

  with torch.no_grad():
    # send to model
    output = model(padded_text[[0], -10:])  # only take the last 10 words (for a balance between context and inference latency)


  output = torch.nn.functional.softmax(output, dim = 1)*100
  op = torch.argsort(output, dim = 1, descending=True)[0, :top_k]
  tokens = []
  for i in op:
    tokens.append(list(vocab.keys())[i])

  return tokens, output[0, op]