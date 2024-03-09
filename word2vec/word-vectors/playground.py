from __future__ import unicode_literals
import spacy 
import numpy as np 
nlp = spacy.load("en_core_web_lg")
doc = nlp(open("joyland.txt").read())
tokens = list(set([w.text.lower() for w in doc if w.is_alpha and w.has_vector]))
sentences = list(doc.sents)

# tokens = []
# with open("joywords.txt", 'r') as file:
#   tokens = file.readlines()
# tokens = [w.strip() for w in tokens]
print(len(tokens))

tokens = list(filter(lambda x: nlp.vocab.has_vector(x), tokens))

def cos_sim(u, v):
  return np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v)

def eucl_dist(u, v):
  return np.linalg.norm(u - v)

def vec(s):
  return np.array(nlp.vocab[s].vector)

# def sentvec(s):
#   sent = nlp(s)
#   vecs = map(lambda x: x.vector, sent)
#   return np.mean(vecs, axis=0)

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

def closest_words(token_list, v, n=10):
  return sorted(token_list, 
                   key=lambda x: cos_sim(vec(x), v), 
                   reverse=True)[:n]

def closest_sents(sents_list, sent, n=10):
  return sorted(sents_list, 
                key=lambda x: cos_sim(x.vector, nlp(sent).vector), 
                reverse=True)[:n]

print(closest_words(tokens, vec("erin")))
# print(nlp("I am very happy").vector)
# for s in closest_sents(sentences, ""):
#   print(s.text)
#   print("---")


