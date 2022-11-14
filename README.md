# mlops-nlp-healthcare-domain

Virtual Assistant Medical Biller 

This is a simple NLP problem where we are trying to predict the ICD code for a given medical biller description. for this task we are using two type of Machine Learning models. 
1) a Classification model that will predict the ICD code for a given medical biller description.
2) a token classification model that will regocnize different diseases in a given medical biller description.

## Dataset

MIMIC-III dataset for ICD Prediction
NC5CDB for Token Classification  

## Pre Trained Models

    Masked language model - BioBert
    BioDischarge Classification - ICD10

## Data Versioning

this project is using DVC to version the data and the models and S3 bucket to mantain the data tracking as well as Huggingface data repository in order to mantain dataset trained for Transformers arquitecture.

## Model Versioning

MlFlow 

## Model Serving

Kubernetes, Triton, Docker and FastAPI

## Model Monitoring

Graphana, Prometheus 

## CI/CD

TBD

