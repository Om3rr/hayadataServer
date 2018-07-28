FROM python:latest
RUN wget https://anaconda.org/pytorch/faiss-cpu/1.2.1/download/linux-64/faiss-cpu-1.2.1-py36_cuda9.0.176_1.tar.bz2
RUN tar xvjf faiss-cpu-1.2.1-py36_cuda9.0.176_1.tar.bz2 && cp -r lib/python3.6/site-packages/* /usr/local/lib/python3.6/site-packages/
RUN pip install mkl
RUN pip install numpy flask
WORKDIR "/app"
CMD python app.py
