import pprint 

# load the [corpus] composed of [documents]
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# frequent words to be removed 
stoplist = set('for a of the and to in'.split(' '))
# lowercase and filter frequent words 
texts = [[w for w in doc.lower().split() if w not in stoplist]
         for doc in text_corpus]

pprint.pprint(texts)
print()

# count word frequencies 
from collections import defaultdict
freq = defaultdict(int)
for txt in texts:
    for token in txt:
        freq[token] += 1

# filter out tokens that only appear once 
processed_corpus = [[token for token in text if freq[token] > 1]
                    for text in texts]

pprint.pprint(processed_corpus)
print()

# dictionary of all the words that we know about 
from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
print()

# print the ids of each token in the dictionary 
pprint.pprint(dictionary.token2id)

# create a bag-of-words vector for a new document (not in the original corpus)
# based on our dictionary defined above
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
print()

# convert the entire corpus to bow vectors 
bow_corpus = [dictionary.doc2bow(w) for w in processed_corpus]
pprint.pprint(bow_corpus)
print()

# convert the bow model of our corpus to a tfidf model, which weighs 
# each token based on its rarity in the corpus, rarer tokens have 
# higher weights 
from gensim import models 
tfidf = models.TfidfModel(bow_corpus)
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])
print()

# transform the whole corpus to tfidf and index it, whatever the fuck that means...
from gensim import similarities
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

# get a ifidf model of the query doc and then compare it to the docs in our 
# corpus to obtain a similarity score with each of them 
query_doc = "system engineering".split()
query_bow = dictionary.doc2bow(query_doc)
sims = index[tfidf[query_bow]]
# make it nice and printable 
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
print()

"""
We saw these concepts in action. 
First, we started with a corpus of documents. 
Next, we transformed these documents to a vector space representation. 
After that, we created a model that transformed our original vector 
representation to TfIdf. Finally, we used our model to calculate the 
similarity between some query document and all documents in the corpus.
"""
