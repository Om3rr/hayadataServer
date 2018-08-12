from flask import Flask, jsonify, request, send_from_directory
from algush import articles, get_idx_by_key, whos_primary, query_by, dont_sort, idxs_to_articles, multiply_vectors, sort_by_distances, search_by_key
import numpy as np
import json
import pdb
app = Flask(__name__, static_url_path='', static_folder='dist', )


@app.route('/api/articles')
def get_atricles():
  return jsonify(articles())

@app.route('/api/multiple')
def query_multi():
  key = request.args.get('idx')
  super_query = True
  mechanism = json.loads(request.args.get('mechanism'))
  purpose = json.loads(request.args.get('purpose'))
  idx = get_idx_by_key(key)
  l, o = whos_primary(purpose, mechanism)
  primary, secondary = l
  prim_o, sec_o = o
  if primary is None:
    return jsonify([])
  N = int(prim_o['slider'])
  distances, idxs = query_by(idx, primary, N)

  #query one vector at a time
  distances = distances[0]
  idxs = idxs[0]
  if(dont_sort(mechanism, purpose)):
    return jsonify(idxs_to_articles(idxs, distances))
  if(prim_o['state'] == sec_o['state']):
    _, idxs_two = query_by(idx, secondary, sec_o['slider'])
    all_idxs = np.unique(np.concatenate([idxs_two[0], idxs]))
    d1 = multiply_vectors(all_idxs, idx, primary)
    d2 = multiply_vectors(all_idxs, idx, secondary)
    final_idxs, final_distances = sort_by_distances(all_idxs, [d1,d2], [1,1])
  else:
    distances_two = multiply_vectors(idxs, idx, secondary)
    final_idxs, final_distances = sort_by_distances(idxs, [distances, distances_two], [prim_o['state'], sec_o['state']])
  return jsonify(idxs_to_articles(final_idxs, final_distances))

@app.route('/api/suggest')
def suggest():
  key = request.args.get('text')
  return jsonify(search_by_key(key))


@app.route('/api/search')
def search():
  key = request.args.get('code')
  idx = get_idx_by_key(key)
  idx = int(idx)
  dist1, idxs1 = query_by(idx, 'abstract', 100, False)
  dist2, idxs2 = query_by(idx, 'title', 100, False)
  dist = np.concatenate([dist1, dist2])
  idxs = np.concatenate([idxs1, idxs2])
  idxs = idxs.reshape((200,))
  dist = dist.reshape((200,))
  zipped = zip(list(idxs), dist)
  sorted_list = sorted(zipped, key=lambda x:x[1])
  s = set()
  final = []
  for idxx, dist in sorted_list:
    if idxx == idx:
      continue
    if dist == 0:
      continue
    if idxx in s:
      continue
    s.add(idxx)
    final.append((idxx, dist))
  idxs, distances = zip(*final)
  idxs = [idx] + list(idxs[0:-1])
  distances = [0] + list(distances[0:-1])
  return jsonify(idxs_to_articles(idxs, distances))


@app.route('/<path:path>')
def static_file(path):
  return app.send_static_file(path)


@app.route('/')
def root():
  return app.send_static_file('index.html')


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
