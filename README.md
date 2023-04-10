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
Input is context words around centered **target** word: [`context <--`, `context <-`, **target**, `context ->`, `context -->`] in form **([context], target)**:
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

Results of **train-nn.txt**, embedded layer_size=30, data.size=1650.
<div align="left">
  <img src="/examples/Figure2.png">
</div>

# Reference

- [NLPTools with text preprocessing](https://github.com/diixo/NLPTools)
- [nlp-starter-continuous-bag-of-words-cbow](https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow)
- [SentEval](https://github.com/diixo/SentEval) by Facebook
- [InferSent](https://github.com/diixo/InferSent) by Facebook
- [Mini-Word2Vec](https://github.com/diixo/MiniWord2Vec)
- [CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model](https://paperswithcode.com/paper/cbow-is-not-all-you-need-combining-cbow-with) with [word2mat](https://github.com/diixo/word2mat)
- [Context encoders as a simple but powerful extension of word2vec](https://paperswithcode.com/paper/context-encoders-as-a-simple-but-powerful)
- [Corrected CBOW Performs as well as Skip-gram](https://paperswithcode.com/paper/koan-a-corrected-cbow-implementation) + Dataset [C4 (Colossal Clean Crawled Corpus)](https://paperswithcode.com/dataset/c4) to [download](https://zenodo.org/record/5542319) for repo: [https://github.com/bloomberg/koan](https://github.com/diixo/koan)
