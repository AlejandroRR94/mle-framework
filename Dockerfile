# FROM ubuntu:22.04
FROM python:3.11.3

# # Update the package lists and install necessary dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libssl-dev \
#     zlib1g-dev \
#     libncurses5-dev \
#     libncursesw5-dev \
#     libreadline-dev \
#     libsqlite3-dev \
#     libgdbm-dev \
#     libdb5.3-dev \
#     libbz2-dev \
#     libexpat1-dev \
#     liblzma-dev \
#     libffi-dev \
#     wget \
#     curl \
#     git \
#     python3-pip

# # Download and extract Python 3.11.3 source code
# RUN apt-get install -y python3

# Set the working directory inside the container
# check our python environment

RUN python3 --version
RUN pip3 --version

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

RUN python pipelines/etl_pipeline.py

RUN python pipelines/train_pipeline.py

# Specify the command to run when the container starts
# CMD ["pytest",  "/test.py"]