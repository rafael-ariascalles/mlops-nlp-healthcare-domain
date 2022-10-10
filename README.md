# mlops-nlp-healthcare-domain

Problem: Medical Biller ICD Code helper

This is a simple NLP problem where we are trying to predict the ICD code for a given medical biller description. for this task we are using two type of Machine Learning models. 
1) a Classification model that will predict the ICD code for a given medical biller description.
2) a token classification model that will regocnize different diseases in a given medical biller description.

## Dataset

the MIMIC-III dataset and the event note will be serve as the biller description.
for the Token Classification model it will be using the NC5CDB and 

## Pre Trained Models

    Masked language model - BioBert
    BioDischarge Classification - ICD9

## Data Versioning

this project is using DVC to version the data and the models and S3 bucket to mantain the data tracking as well as Huggingface data repository in order to mantain dataset trained for Transformers arquitecture.

## Model Versioning

TBD

## Model Serving

Kubernetes, Docker and FastAPI

## Model Monitoring

Graphana, Prometheus and Loki (TBD)


## CI/CD

TBD

