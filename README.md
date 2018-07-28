this is the server side of hayadata. this one should dockerize (using Dockerfile)


to install env first you need to get "data.pkl" its a big file (~111MB) that you can ask from Omer (or can take it from the server its already there!)

after you got this huge file, put it here (under hayadataServer)

after that you can run (you should be cd`ed inside this folder and have data.pkl after you builded the Dockerimage using
docker build -t hayadata .
docker run -it --mount type=bind,source="$(pwd)",target=/app -p 5001:5000 -h 0.0.0.0 hayadata

in algush there some data preparation and all the data functions are defined there while the api endpoints are defined under app.py


the data from data.pkl looks like


data.pkl = {'data': data, 'key_to_idx': key_to_idx}

data = {'title': [.....] (list), 'title vector': [......], 'abstract': [...], 'abstract vector': [....]}
while title[1] related to 'title vector'[1] and so on

key_to_idx = {key: idx} to fetch elements from data

im using faiss to get elements
