from __future__ import unicode_literals
import spacy 
import numpy as np 

# charge les vecteurs dans notre modèle (nlp) 
nlp = spacy.load("en_core_web_lg")
# lit le texte (joyland.txt) et crée ainsi notre corpus (doc) 
doc = nlp(open("joyland.txt").read())

# garde que les mots alphabétiques et les convertit en minuscule 
tokens = list(set([w.text.lower() for w in doc if w.is_alpha]))
# garde que les mots qui ont un vecteur dans notre modèle 
tokens = list(filter(lambda x: nlp.vocab.has_vector(x), tokens))
# extrait les phrases 
sentences = list(doc.sents)

""" Calcule la similarité cosinus entre deux vecteurs. """
def cos_sim(u, v):
  return np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v)

""" Retourne le vecteur correspondant au mot donné en entrée. """
def vec(s : str):
  return np.array(nlp.vocab[s].vector)

""" Retourne le mot correspondant au vecteur donné en entrée, 
en cherchant dans le modèle le vecteur qui est le plus près de
l'entrée. """
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

""" Retourne la liste des mots les plus similaires au mot en entrée. """
def closest_words(token_list, v, n=5):
  return sorted(token_list, 
                   key=lambda x: cos_sim(vec(x), v), 
                   reverse=True)[:n]

""" Retorne la liste des phrases les plus similaires à la phrase en entrée. """
def closest_sents(sents_list, sent, n=5):
  return sorted(sents_list, 
                key=lambda x: cos_sim(x.vector, nlp(sent).vector), 
                reverse=True)[:n]

#print(closest_words(tokens, vec("food")))
#for s in closest_sents(sentences, "Not knowing what to do"):
#  print(s.text)
#  print("---")


