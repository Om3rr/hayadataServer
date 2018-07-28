import pickle
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

f = open('db.pickle', 'rb')
raw_data = pickle.load(f)
title = []
title_keys = []
a = []
a_keys = []
for k, v in list(raw_data['abstract'].items()):
  title.append(v)
  title_keys.append(k)
for k, v in list(raw_data['title'].items()):
  a.append(v)
  a_keys.append(k)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', ngram_range=(1, 3),
                                   use_idf=True)
tfidf_vectorizer.fit(title + a)
print("Fit!")

terms = tfidf_vectorizer.get_feature_names()
d = {}
t = {}
print("Transform!")
for row, key in zip(a, a_keys):
  v = tfidf_vectorizer.transform([row]).toarray()
  v = v[0]
  best_args = v.argsort()[-10:][::-1]
  best_terms = [terms[i] for i in best_args]
  d[key] = best_terms
print("Transform2!")
for row, key in zip(title, title_keys):
  v = tfidf_vectorizer.transform([row]).toarray()
  v = v[0]
  best_args = v.argsort()[-10:][::-1]
  best_terms = [terms[i] for i in best_args]
  t[key] = best_terms
print("Save")
pickle.dump({"titles": t, "abstracts": d}, 'tf-idf.txt')

# for i, comp in enumerate(svd.components_):
# for i, comp in enumerate(matrix.asarray()):
#	tt = zip(terms, comp)
#	sorted_terms = sorted(tt, key=lambda x:x[1], reverse=True)[:10]
#	print(sorted_terms)
##	print((title+a)[i])
