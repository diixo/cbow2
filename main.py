
# https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow/

import re
import numpy as np
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from subprocess import check_output

# - the word embeddings as inputs (idx)
# - the linear model as the hidden layer
# - the log_softmax as the output

class Sentencizer: #from NLPTools

    def __init__(self, input_text, split_characters=['.','?','!',':'], delimiter_token='<split>'):
        self.sentences = []
        self.raw = str(input_text)
        self._split_characters = split_characters
        self._delimiter_token = delimiter_token
        self._index = 0
        self._stopwords = [line.replace('\n', '') for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]
        self.vocab = set()
        self._sentencize()

    def _sentencize(self):
        work_sentence = self.raw
        for character in self._split_characters:
            work_sentence = work_sentence.replace("\n",  " ")
            work_sentence = work_sentence.replace(character, character + "" + self._delimiter_token)
        self.sentences = [x.strip().lower() for x in work_sentence.split(self._delimiter_token) if x !='']

        work_sentence = ""
        punctuations = string.punctuation
        token_boundaries = [' ', ',', '.']


        for i in range(len(self.sentences)):
            for punctuation in punctuations:
                work_sentence = self.sentences[i].replace(punctuation, " " + punctuation + " ")

            for delimiter in token_boundaries:
                work_sentence = work_sentence.replace(delimiter, self._delimiter_token)

            self.sentences[i] = [x.strip() for x in work_sentence.split(self._delimiter_token) if (x != '' and x not in self._stopwords)]
            self.vocab.update(set(self.sentences[i]))
            #print(self.sentences[i])
        #print(self.vocab)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.sentences):
            result = self.sentences[self._index]
            self._index += 1
            return result
        raise StopIteration


sentences = """We are about to study the idea of computational process.
 Computational processes are abstract beings that inhabit computers.
 As they evolve, processes manipulate other abstract things called data.
 The evolution of a process is directed by a pattern of rules
 called a program. People create programs to direct processes. In effect,
 we conjure the spirits of the computer with our spells."""

tokenizer = Sentencizer(sentences)

epochs = 100
vocab_size = len(tokenizer.vocab)
embed_dim = 10
context_size = 2

word_to_ix = {word: i for i, word in enumerate(tokenizer.vocab)}
ix_to_word = {i: word for i, word in enumerate(tokenizer.vocab)}

# data - [(context), target]
data = []

for sentence in tokenizer.sentences:
    for i in range(2, len(sentence) - 2):
        context = [sentence[i - 2], sentence[i - 1], sentence[i + 1], sentence[i + 2]]
        target = sentence[i]
        data.append((context, target))
print(data[:10])

# Embeddings
embeddings = np.random.random_sample((vocab_size, embed_dim))

# Linear-model
def linear(m, theta):
    w = theta
    return m.dot(w)

# Log softmax + NLLloss = Cross Entropy
def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)


def log_softmax_crossentropy_with_logits(logits, target):
    out = np.zeros_like(logits)
    out[np.arange(len(logits)), target] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- out + softmax) / logits.shape[0]

# Forward propagation
def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)

    return m, n, o

# Backward propagation
def backward(preds, theta, target_idxs):
    m, n, o = preds

    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)

    return dw

def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta

# Training:
theta = np.random.uniform(-1, 1, (2 * context_size * embed_dim, vocab_size))

epoch_losses = {}

for epoch in range(epochs):

    losses = []

    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)

        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)

    epoch_losses[epoch] = losses

# Analyze: Plot loss/epoch
def analyze():
    ix = np.arange(0, epochs)
    fig = plt.figure()
    fig.suptitle('Epoch/Losses', fontsize=20)
    plt.plot(ix,[epoch_losses[i][0] for i in ix])
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Losses', fontsize=12)
    fig.show()


def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]

    return word

analyze()

# (['evolve', 'processes', 'abstract', 'things'], 'manipulate')
w = predict(['evolve', 'processes', 'abstract', 'things'])
print(w)


def accuracy():
    wrong = 0

    for context, target in data:
        if (predict(context) != target):
            wrong += 1

    return (1 - (wrong / len(data)))

print(accuracy())

