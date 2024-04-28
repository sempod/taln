from __future__ import unicode_literals
import spacy 
import numpy as np 

# load the word vectors into our model 
nlp = spacy.load("en_core_web_lg")
# read the text (joyland.txt) and create our corpus (doc)
doc = nlp(open("joyland.txt").read())

# keep only alphabetical words and lowercase them 
tokens = list(set([w.text.lower() for w in doc if w.is_alpha]))
# keep only words that have a vector in our model 
tokens = list(filter(lambda x: nlp.vocab.has_vector(x), tokens))
# extract sentences
sentences = list(doc.sents)

""" Calculates the cosine similarity between two vectors. """
def cos_sim(u, v):
  return np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v)

""" Returns the vector corresponding to the input string. """
def vec(s : str):
  return np.array(nlp.vocab[s].vector)

""" Return the word corresponding to the given vector by looking 
for the closest vector in the model.
"""
def word(v):
  max_sim = -1 
  res = ""
  for w in nlp.vocab:
    if w.has_vector:
      sim = cos_sim(v, w.vector)
      if sim > max_sim:
        max_sim = sim 
        res = w.text
  return res  

""" Return the list of most similar words to the given one. """
def closest_words(token_list, v, n=5):
  return sorted(token_list, 
                   key=lambda x: cos_sim(vec(x), v), 
                   reverse=True)[:n]

""" Return closest sentences """
def closest_sents(sents_list, sent, n=5):
  return sorted(sents_list, 
                key=lambda x: cos_sim(x.vector, nlp(sent).vector), 
                reverse=True)[:n]

#print(closest_words(tokens, vec("food")))
#for s in closest_sents(sentences, "Not knowing what to do"):
#  print(s.text)
#  print("---")


