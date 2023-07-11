# Use a base image with the desired operating system and installed dependencies
FROM python:3.11

# Set the working directory inside the container
WORKDIR /

# Copy the code into the container
COPY . .

# Install dependencies
RUN pip install -r /requirements.txt

# Specify the command to run when the container starts
CMD ["pytest",  "/test.py"]
