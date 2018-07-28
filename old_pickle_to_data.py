import pickle
from functools import reduce
import numpy as np
f = open('db.pickle', 'rb')
raw_data = pickle.load(f)
data = {}
keyz = ['abstract', 'title', 'title vector', 'abstract vector']
key_to_idx = {x: idx for idx, x in enumerate(set(reduce(lambda x, y: x + y, [list(raw_data[x].keys()) for x in keyz])))}
idx_to_key = {v: k for k, v in key_to_idx.items()}
for k in raw_data:
  data[k] = [0] * len(key_to_idx)
  unspecified_list = [x if x not in raw_data[k] else None for x in key_to_idx.keys()]
  for kk in raw_data[k]:
    idx = key_to_idx[kk]
    data[k][idx] = raw_data[k][kk]
  for kk in unspecified_list:
    if kk is None:
      continue
    idx = key_to_idx[kk]
    if k.endswith('vector'):
      data[k][idx] = np.zeros(300)
    else:
      data[k][idx] = 'Empty'

new_data = {'data': data, 'key_to_idx': key_to_idx}
fw = open('data.pkl', 'wb')
pickle.dump(new_data, fw)
