# FROM ubuntu:22.04
FROM python:3.11.3

# 
RUN mkdir app

WORKDIR /app

# Copy the code into the container

# RUN cd app

# RUN cat /etc/os-release > info.txt

COPY . .
# RUN mv .. .
# RUN mkdir data/clean data/raw data/test
RUN mkdir data/clean data/raw data/database

# Install dependencies
RUN pip install -r requirements.txt

# RUN python pipelines/etl_pipeline.py

# RUN python pipelines/train_pipeline.py

# Specify the command to run when the container starts
CMD ["pytest",  "test.py"]