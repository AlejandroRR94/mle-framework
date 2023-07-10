# Use a base image with the desired operating system and installed dependencies
FROM python:3.11.3

# Set the working directory inside the container
WORKDIR /app

# Copy the code into the container
COPY * app/

# Install dependencies
RUN pip install -r app/requirements.txt

# Specify the command to run when the container starts
CMD ["pytest", "tests.py"]
