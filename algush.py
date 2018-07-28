import faiss
import pickle
import numpy as np
import random

ARTICLES_PER_PAGE = 102


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


def query_by(idx, by='abstract', N=1000):
  """
  wrapper to query function to make an easier query
  :param idx: idx of the patent
  :param by: abstract\title
  :param N: Size of array to return
  :return:
  """
  to_query = index_titles if by == 'title' else index_abstract
  vectors = title_vectors if by == 'title' else abstract_vectors
  return to_query.search(np.asarray([vectors[idx]]), N)


def idxs_to_articles(idxs, distances):
  """
  map a list of idxs to patents representations
  :param idxs: a list of integers
  :param distances: distance for each idx
  :return: a list of articles
  """
  articles = []
  for d, s in zip(distances, idxs):
    articles.append({"title": data['title'][s], "abstract": data['abstract'][s], "index": int(s), "distance": int(d), "key": idx_to_key[s]})
  return articles

def get_idx_by_key(key):
  """
  wrapper to key_to_idx dict
  :param key: patent key
  :return: patent idx
  """
  if (key not in key_to_idx):
    return None
  return key_to_idx[key]

def multiply_vectors(idxs, idx, by):
  """
  Calculate L2 norm of idxs against idx
  :param idxs: a list of integeres that represent patents
  :param idx: a integere of patent that we want to compare
  :param by: title\abstract
  :return: a list of distances which each distance represent the idx in the same position
  """
  vectors = title_vectors if by == 'title' else abstract_vectors
  vector = vectors[idx]
  vs = np.take(vectors, idxs, axis=0) - vector
  vs = vs**2
  mul = np.sum(vs, axis=1)
  return np.sqrt(mul)

def sort_by_distances(idxs, distances, states):
  """
  a function that sort the given indexes by distances and states
  :param idxs: a list of patents
  :param distances: list of lists distances[0][idx] represent the first distance of patent idx
  :param states: represent the state of each filter (0 - doesnt matter, 1 - similar, 2-dissimilar)
  :return: a tuple of idxs, dist (sorted)
  """
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
  """
  predictaor to check if we want to sort or not
  :param p: purpose object
  :param m: mechanism object
  :return: to sort or not.
  """
  return p['state'] * m['state'] == 0

def whos_primary(p, m):
  """
  predicator to check prioritize of purpose and mechanism
  :param p: purpose object
  :param m: mechanism object
  :return: [titleA, titleB], [objA, objB]
  """
  if p['state'] == 1:
    return ['title', 'abstract'], [p, m]
  if m['state'] == 1:
    return ['abstract', 'title'], [m, p]
  return None, None


def search_by_key(key):
  """
  For patent suggestion, function that search in the data internally
  :param key: free text of the user.
  :return: a list of relevant patent, each patent will look like {key: key, title: title}
  """
  okays = []
  key = key.lower()
  for idx, title in enumerate(data['title']):
    if len(okays) > 20:
      return {'response': okays}
    code = idx_to_key[idx]
    if key in code.lower() or key in title.lower():
      okays.append({'code': code, 'title': title})
  return {'response': okays}
