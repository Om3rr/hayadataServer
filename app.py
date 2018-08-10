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
  super_query = request.args.get('super')
  mechanism = json.loads(request.args.get('mechanism'))
  purpose = json.loads(request.args.get('purpose'))
  idx = get_idx_by_key(key)
  l, o = whos_primary(purpose, mechanism)
  primary, secondary = l
  prim_o, sec_o = o
  if primary is None:
    return jsonify([])
  N = int(prim_o['slider'])
  print("Super? ", super_query)
  distances, idxs = query_by(idx, primary, N, super_query)

  #query one vector at a time
  distances = distances[0]
  idxs = idxs[0]
  if(dont_sort(mechanism, purpose)):
    return jsonify(idxs_to_articles(idxs, distances))
  if(prim_o['state'] == sec_o['state']):
    _, idxs_two = query_by(idx, secondary, sec_o['slider'], super_query)
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

@app.route('/<path:path>')
def static_file(path):
  return app.send_static_file(path)


@app.route('/')
def root():
  return app.send_static_file('index.html')


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
