# Word2Vec
2 new model architectures for **continuous vector representation(?)**.
Their quality is measured in word similarity tasks and compared to existing models. 
More accurate and trained in less time. 
Less than a day to learn vectors from a dataset of 1.6 billion words - wasn't possible before.

## Introduction 
*"Many current NLP models (current being 2013) treat words are **atomic units(?)** - 
there's no notion of similarity."*
Such as the **N-gram(?)** model used for statistical lang. modeling. This model works great
when trained ona huge dataset. However, for some NLP tasks, the datasets are of limited
size: speech recognition, translation, etc. For this, we need more complex models. 
*"Probably the most successful concept is to use **distributed representations of words(?)**"*

## Goal 
Technique to learn vectors from a dataset of billions of words. 
**Multiple degrees of similarity**, for example in case of inflectional languages, such 
as Czech or Slovak, where the same noun can have different endings - these words should 
be close to each other. 
Preserving **linear regularities** between words: king - man + woman = queen. 

## Previous work
Just mention that this paper didn't invent the idea of representing words as vectors, 
and models already existed for this. 
They define the complexity of a model as O = E * T * Q, where E is the number of 
**training epochs(?)**, T is the number of words in the training set and Q is defined
further for each model. 

## New Log-linear Models
*"The new architectures directly follow those proposed in our earlier work [13, 14], where it was
found that neural network language model can be successfully trained in two steps: ﬁrst, continuous
word vectors are learned using simple model, and then the N-gram NNLM is trained on top of these
distributed representations of words.**(?)**"*
They all use **stochastic gradient descent** and **backpropagation**.

### CBOW 
Similar to the **feedforward NNLM(?)**. 
The non-linear hidden layer is removed and the **projection layer(?)** is shared for all 
words. 
The order of words in the history doesn't influence the projection. 
They build a log-linear classifier with 4 future and 4 history words at the input.
The training criterion is to correctly classify the current (middle) word. 

### Continuous Skip-gram 
Similar to CBOW, however here we have the current word as input to a log-linear classifier
with a **continuous projection layer**, and we try to predict words within a certain range 
before and after the current one. 
*"We found that increasing the range improves quality of the resulting word vectors, but it also increases
the computational complexity. Since the more distant words are usually less related to the current
word than those close to it, we give less weight to the distant words by sampling less from those
words in our training examples."*

## Results 
Previous papers typically use a table showing similar words to a chosen one. 
Showing that the word *France* is similar to the word *Italy* is cool, but this paper
tries to go further than that. 
*We follow previous observation that there can be many different types of similarities between words, for
example, word big is similar to bigger in the same sense that small is similar to smaller. Example
of another type of relationship can be word pairs big - biggest and small - smallest [20]. We further
denote two pairs of words with the same relationship as a question, as we can ask: ”What is the
word that is similar to small in the same sense as biggest is similar to big?”*

*Somewhat surprisingly, these questions can be answered by performing simple algebraic operations
with the vector representation of words. To ﬁnd a word that is similar to small in the same sense as
biggest is similar to big, we can simply compute vector X = vector(”biggest”) − vector(”big”) +
vector(”small”). Then, we search in the vector space for the word closest to X measured by cosine
distance, and use it as the answer to the question (we discard the input question words during this
search). When the word vectors are well trained, it is possible to ﬁnd the correct answer (word
smallest) using this method.*

*Finally, we found that when we train high dimensional word vectors on a large amount of data, the
resulting vectors can be used to answer very subtle semantic relationships between words, such as
a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin. Word vectors
with such semantic relationships could be used to improve many existing NLP applications, such
as machine translation, information retrieval and question answering systems, and may enable other
future applications yet to be invented.*

## Task description 
Test set containing 5 types of semantic questions and 9 types of syntactic questions. 
Overall, 8869 semantic and 10675 syntactic questions. 

Question is correctly answered iff the closest word to the computed vector is that of 
the correct answer. Synonyms don't count, so reaching 100% is pretty much impossible. 

*"Further progress can be achieved by incorporating information
about structure of words, especially for the syntactic questions."*

## Maximization of accuracy 
They used a Google News corpus - 6B tokens. Restricted the vocab to 1 million most 
frequent words. Basically we need both more data and more dimensions to improve 
accuracy.

## Comparison of model architectures 
Different models using the same training data and the same 640 dimensions. 
Full set of questions from the mentioned **Semantic-Syntactic Word Relationship test set**.

**Show Table 3 here and talk a bit about the other models**

**Show Table 4 - publicly available word vectors compared to these**

**Maybe Table 6 as well**

## Examples of learned relationships 
Another way to improve accuracy is to provide more than one example of a given 
relationship - you average the individual vectors to form the relationship
vector - 10% accuracy increase on the semantic-syntactic test.

You can also use it to find a word that doesn't belong to a list of words.
Compute the average of the list and then pick the most distant word. 
Popular in certain human intelligence tests. 

## Conclusion
Ok so the NNLM and RNLM models are **neural networks**, because they have a big hidden 
layer, and the CBOW and Skip-gram models proposed here aren't, since they have a small 
hidden layer and are quite simple. Yet they achieve better results. 




