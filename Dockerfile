FROM python:3.6
ENV PYTHONUNBUFFERED 1
RUN mkdir /facenet
COPY . /facenet
RUN pip install -r /facenet/new_requirements.txt
WORKDIR /facenet/src