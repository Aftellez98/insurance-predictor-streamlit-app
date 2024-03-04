# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the app files to the container
COPY main.py .
COPY data /app/data
COPY notebooks /app/notebooks
COPY requirements.txt .

# Install the app dependencies
RUN pip install -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
