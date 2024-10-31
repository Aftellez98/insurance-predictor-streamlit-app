# Insurance Predictor App

Welcome to the Insurance Predictor App! This application leverages a simple supervised machine learning model (LASSO) to predict insurance charges based on user input. It's designed with Streamlit, making it user-friendly and easy to interact with.

## Project Overview

The Insurance Predictor App uses historical insurance data to train a supervised machine learning model. The model learns to predict insurance charges based on features such as age, sex, BMI, number of children, smoking status, and region. The goal is to provide users with an estimate of their insurance costs based on their input parameters.

## Getting Started

### **Running App Locally
To run the Insurance Predictor App on your local machine, you'll need to have Python and Streamlit installed:

```sh
pip install streamlit
streamlit run main.py
```

### **Running App Locally

```sh
docker build -t my-python-app .
docker run -p 8501:8501 my-python-app
docker run -p 8501:8501 --env-file env.list my-python-app
```