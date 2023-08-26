FROM python:3.11.4

RUN mkdir /fastapi-app

WORKDIR /fastapi-app

COPY requirements.txt /fastapi-app

RUN pip install -r requirements.txt

COPY . /fastapi-app
