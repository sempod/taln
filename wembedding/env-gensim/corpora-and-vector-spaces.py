import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docs = [
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

from pprint import pprint
from collections import defaultdict

stopset = set('for a of the and to in'.split())
docslisted = [[w for w in doc.lower().split() if w not in stopset]
         for doc in docs]

freq = defaultdict(int)
for doc in docslisted:
    for token in doc:
        freq[token] += 1
docslisted = [[token for token in doc if freq[token] > 1]
              for doc in docslisted]
pprint(docslisted)

from gensim import corpora
dictionary = corpora.Dictionary(docslisted)
dictionary.save("/tmp/sample.dict")
print(dictionary)
print()

print(dictionary.token2id)
print()

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
print()

corpus = [dictionary.doc2bow(doc) for doc in docslisted]
corpora.MmCorpus.serialize("/tmp/sample.mm", corpus)
print(corpus)
print()

from smart_open import open 
class MyCorpus:
    # custom iter function for this specific corpus 
    def __iter__(self):
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

"""
The full power of Gensim comes from the fact that a corpus doesn’t 
have to be a list, or a NumPy array, or a Pandas dataframe, or whatever. 
Gensim accepts any object that, when iterated over, successively yields documents.
"""

corpus_memfriendly = MyCorpus()
print(corpus_memfriendly)
print()

# print the individual vectors of this corpus 
for vec in corpus_memfriendly:
    print(vec)
print()

