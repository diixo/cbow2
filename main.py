
# https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow/

import re
import numpy as np
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter

from subprocess import check_output

# - the word embeddings as inputs (idx)
# - the linear model as the hidden layer
# - the log_softmax as the output

class Sentencizer: #from NLPTools

    def __init__(self, split_characters=['.', '?', '!', ':', ';', ','], delimiter_token='<split>'):
        self.sentences = []
        self._split_characters = split_characters
        self._delimiter_token = delimiter_token
        self._index = 0
        self._stopwords = [line.replace('\n', '') for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]
        self.vocab = set()
        self.vocab_freq = {}
        self.vocab_freq_sorted = {}

    def sentencize(self, input_line):
        work_sentence = input_line.strip()
        sentences = []

        if (work_sentence == ""):
            return

        for character in self._split_characters:
            work_sentence = work_sentence.replace("\n",  " ")
            work_sentence = work_sentence.replace(character, character + "" + self._delimiter_token)

        sentences = [x.strip().lower() for x in work_sentence.split(self._delimiter_token) if x !='']

        token_boundaries = [' ', ',', '.']

        for i in range(len(sentences)):
            work_sentence = sentences[i]

            for delimiter in token_boundaries:
                work_sentence = work_sentence.replace(delimiter, self._delimiter_token)

            sentences[i] = [x.strip() for x in work_sentence.split(self._delimiter_token) if (x != '')]

            work_sentence = []
            for w in sentences[i]:
                w = w.strip(string.punctuation)
                if (w != '' and w not in self._stopwords and not w.isdigit()):
                    work_sentence.append(w)

                    if w in self.vocab_freq:
                        self.vocab_freq[w] += 1
                        continue
                        # print (word, vocab[word])
                    self.vocab_freq[w] = 1

            if (len(work_sentence) > 0):
                #print(' '.join(sentences[i]))
                #print(' '.join(work_sentence))
                self.sentences.append(work_sentence)
                self.vocab.update(set(work_sentence))
        #print(self.vocab)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.sentences):
            result = self.sentences[self._index]
            self._index += 1
            return result
        raise StopIteration

    def readFile(self, filename):

        f = open(filename, 'r', encoding='utf-8')
        count = 0;
        while True:
            line = f.readline()
            if not line:
                break;

            count+=1
            self.sentencize(line)

        f.close();
        self.vocab = sorted(self.vocab)
        self.vocab_freq_sorted = sorted(self.vocab_freq.items(), key=itemgetter(1), reverse=True)

sentences = """We are about to study the idea of computational process.
 Computational processes are abstract beings that inhabit computers.
 As they evolve, processes manipulate other abstract things called data.
 The evolution of a process is directed by a pattern of rules
 called a program. People create programs to direct processes. In effect,
 we conjure the spirits of the computer with our spells."""

tokenizer = Sentencizer()
#tokenizer.sentencize(sentences)
tokenizer.readFile("train-nn.txt")
#print(tokenizer.vocab_freq_sorted)

epochs = 100
vocab_size = len(tokenizer.vocab)
embed_dim = 100  #sqrt(tokenizer.sentences.sz)
context_wnd = 3 # 2, 3 or 4: [(context_wnd), target]

word_to_ix = {word: i for i, word in enumerate(tokenizer.vocab)}
ix_to_word = {i: word for i, word in enumerate(tokenizer.vocab)}

# data - [(context), target]
data = []

############
for sentence in tokenizer.sentences:
    if (context_wnd == 4):
        for i in range(2, len(sentence) - 2):
            context = [sentence[i - 2], sentence[i - 1], sentence[i + 1], sentence[i + 2]]
            target = sentence[i]
            data.append((context, target))

    if (context_wnd == 3):
        for i in range(0, len(sentence) - 3):
            context = [sentence[i], sentence[i + 1], sentence[i + 2]]
            target = sentence[i + 3]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 1], sentence[i + 3]]
            target = sentence[i + 2]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 2], sentence[i + 3]]
            target = sentence[i + 1]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

    if (context_wnd == 2):
        for i in range(0, len(sentence) - 2):
            context = [sentence[i], sentence[i + 1]]
            target = sentence[i + 2]
            data.append((context, target))
            #print("#" + target + " : " + sentence[i + 1] + ", " + sentence[i + 2])

            if (i == 10000):
                context = [sentence[i + 1], sentence[i + 2]]
                target = sentence[i]
                data.append((context, target))
                #print("#" + target + " : " + sentence[i] + ", " + sentence[i + 1])
############
print(str(len(data)))
#print(str(len(tokenizer.sentences)))

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

    return (-out + softmax) / logits.shape[0]

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

def optimize(theta, grad, lr = 0.03):
    theta -= grad * lr
    return theta

# Training:
theta = np.random.uniform(-1, 1, (context_wnd * embed_dim, vocab_size))

epoch_losses = {}
success = []

for epoch in range(epochs):
    losses = []
    hits = 0

    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)

        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)

        ##########################################
        word_id = np.argmax(preds[-1])
        if (word_id == word_to_ix[target]): hits += 1
        ##########################################

    epoch_losses[epoch] = losses
    success.append(hits/len(data)*100.0)
    print("<< " + str(epoch) + " : " + str(hits/len(data)*100.0))

# Analyze: Plot loss/epoch
def plot_loss():
    ix = np.arange(0, epochs)
    fig = plt.figure()
    fig.suptitle('Epoch/Losses', fontsize=20)
    plt.plot(ix,[epoch_losses[i][0] for i in ix])
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Losses', fontsize=12)
    fig.show()

def plot_precision():
    ix = np.arange(0, epochs)
    fig = plt.figure()
    fig.suptitle('Epoch/Precision%, '+str(int(success[epochs-1])), fontsize=20)
    plt.plot(ix,[success[i] for i in ix])
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Precision,%', fontsize=12)
    fig.show()

def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]
    return word

def verify():
    sz = len(data)
    success = 0;
    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)
        word_id = np.argmax(preds[-1])
        word = ix_to_word[word_id]
        target_id = word_to_ix[target]
        if (word_id == target_id) : success += 1
    print("sucess:", str(100.0 * success/sz))

def accuracy():
    wrong = 0
    for context, target in data:
        if (predict(context) != target):
            wrong += 1
    return (1 - (wrong / len(data)))

plot_loss()
plot_precision()

verify()

# (['evolve', 'processes', 'abstract', 'things'], 'manipulate')
w = predict(['evolve', 'processes', 'abstract', 'things'])
print(w)

print(accuracy())
