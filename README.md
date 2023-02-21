# cbow2
### CBOW

**CBOW** or **Continous bag of words** is to use embeddings in order to train a neural network where the context is represented by multiple words for a given target words.

For example, we could use “cat” and “tree” as context words for “climbed” as the target word.
This calls for a modification to the neural network architecture.
The modification, shown below, consists of replicating the input to hidden layer connections C times, the number of context words, and adding a divide by C operation in the hidden layer neurons.

The CBOW architecture is pretty simple contains :

- the word embeddings as inputs (idx)
- the linear model as the hidden layer
- the log_softmax as the output

### Input:
```
sentence = "we are about to study the idea of computational"
```
Input dataset is context words around key-centered target: [`context <--`, `context <-`, **target**, `context ->`, `context -->`] in form **([context], target)**:
```
[(['we', 'are', 'to', 'study'], 'about'), (['are', 'about', 'study', 'the'], 'to'), (['about', 'to', 'the', 'idea'], 'study'), (['to', 'study', 'idea', 'of'], 'the'), (['study', 'the', 'of', 'computational'], 'idea')]
```

Trained verification input: 
```python
# (['we', 'are', 'to', 'study'], 'about')
word = predict(['we', 'are', 'to', 'study'])
```

### Output: 
**word** =`'about'`

# Reference

[nlp-starter-continuous-bag-of-words-cbow](https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow)
