
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

sentences = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

# remove special characters
sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)

# remove 1 letter words
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()

# lower all characters
sentences = sentences.lower()

words = sentences.split()
vocab = set(words)

epochs = 100
vocab_size = len(vocab)
embed_dim = 10
context_size = 2

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# data - [(context), target]
data = []
for i in range(2, len(words) - 2):
    context = [words[i - 2], words[i - 1], words[i + 1], words[i + 2]]
    target = words[i]
    data.append((context, target))
print(data[:5])

# Embeddings
embeddings =  np.random.random_sample((vocab_size, embed_dim))

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

# (['we', 'are', 'to', 'study'], 'about')
w = predict(['we', 'are', 'to', 'study'])
print(w)


def accuracy():
    wrong = 0

    for context, target in data:
        if (predict(context) != target):
            wrong += 1

    return (1 - (wrong / len(data)))

print(accuracy())

