import faiss
import pickle
import numpy as np
import random

ARTICLES_PER_PAGE = 102

#
# f = open('db.pickle', 'rb')
# raw_data = pickle.load(f)
# data = {}
# keyz = ['abstract', 'title', 'title vector', 'abstract vector']
# key_to_idx = {x: idx for idx, x in enumerate(set(reduce(lambda x, y: x + y, [list(raw_data[x].keys()) for x in keyz])))}
# idx_to_key = {v: k for k, v in key_to_idx.items()}
# for k in raw_data:
#   data[k] = [0] * len(key_to_idx)
#   unspecified_list = [x if x not in raw_data[k] else None for x in key_to_idx.keys()]
#   for kk in raw_data[k]:
#     idx = key_to_idx[kk]
#     data[k][idx] = raw_data[k][kk]
#   for kk in unspecified_list:
#     if kk is None:
#       continue
#     idx = key_to_idx[kk]
#     if k.endswith('vector'):
#       data[k][idx] = np.zeros(300)
#     else:
#       data[k][idx] = 'Empty'
#
# new_data = {'data': data, 'key_to_idx': key_to_idx}
# fw = open('data.pkl', 'wb')
# pickle.dump(new_data, fw)
fw = open('data.pkl', 'rb')
raw_data = pickle.load(fw)

data = raw_data['data']
key_to_idx = raw_data['key_to_idx']
idx_to_key = {v: k for k, v in key_to_idx.items()}

m = min(len(data['abstract vector']), len(data['title vector']))
abstract_vectors = np.asarray(data['abstract vector'], dtype=np.float32)
title_vectors = np.asarray(data['title vector'], dtype=np.float32)

_, d = abstract_vectors.shape
nq = 10000
index_abstract = faiss.IndexFlatL2(d)
index_abstract.add(abstract_vectors)
_, d = title_vectors.shape
index_titles = faiss.IndexFlatL2(d)
index_titles.add(title_vectors)


def articles():
  samp = random.sample(range(0, len(data['abstract'])), ARTICLES_PER_PAGE)
  return list(
    map(lambda x: {"title": data['title'][x], "abstract": data['abstract'][x], "index": x, "key": idx_to_key[x]}, samp))


def query(idx, by='abstract'):
  idx = int(idx)
  to_query = index_titles if by == 'title' else index_abstract
  vectors = title_vectors if by == 'title' else abstract_vectors
  dist, scores = to_query.search(np.asarray([vectors[idx]]), ARTICLES_PER_PAGE)
  articles = []
  scores = scores[0]
  dist = dist[0]
  for d, s in zip(dist, scores):
    articles.append({"title": data['title'][s], "abstract": data['abstract'][s], "index": int(s), "distance": int(d)})
  return articles


def query_by(idx, by='abstract', N=1000):
  to_query = index_titles if by == 'title' else index_abstract
  vectors = title_vectors if by == 'title' else abstract_vectors
  return to_query.search(np.asarray([vectors[idx]]), N)


def idxs_to_articles(idxs, distances):
  articles = []
  for d, s in zip(distances, idxs):
    articles.append({"title": data['title'][s], "abstract": data['abstract'][s], "index": int(s), "distance": int(d), "key": idx_to_key[s]})
  return articles

def get_idx_by_key(key):
  if (key not in key_to_idx):
    return None
  return key_to_idx[key]

def multiply_vectors(idxs, idx, by):
  vectors = title_vectors if by == 'title' else abstract_vectors
  vector = vectors[idx]
  vs = np.take(vectors, idxs, axis=0) - vector
  vs = vs**2
  mul = np.sum(vs, axis=1)
  return np.sqrt(mul)

def sort_by_distances(idxs, distances, states):
  d1, d2 = distances
  s1, s2 = states
  idxs = list(idxs)
  reversed = False
  if (s1 == 2):
    d = d1
    reversed = True
  elif (s2 == 2):
    d = d2
    reversed = True
  else:
    #it means that s1 == s2 == 1
    d = np.maximum(d1, d2)
  argsort = d.argsort()
  if reversed:
    argsort = np.flip(argsort, 0)[:-1]
    argsort = np.insert(argsort, 0, 0)
  idxs = np.take(idxs, argsort)
  dist = np.take(d, argsort)
  return idxs, dist

def dont_sort(p, m):
  return p['state'] * m['state'] == 0

def whos_primary(p, m):
  if p['state'] == 1:
    return ['title', 'abstract'], [p, m]
  if m['state'] == 1:
    return ['abstract', 'title'], [m, p]
  return None, None


def search_by_key(key):
  okays = []
  key = key.lower()
  for idx, title in enumerate(data['title']):
    if len(okays) > 20:
      return {'response': okays}
    code = idx_to_key[idx]
    if key in code.lower() or key in title.lower():
      okays.append({'code': code, 'title': title})
  return {'response': okays}
