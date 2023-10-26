This repository contains:
* Code to perform exploratory data analysis on and clean the UCI Machine Learning Repository Cenus Data dataset
* Utility functions and a script to train a Random Forest Classifier model to classify whether an individual makes more or less than $50,000 per year based on their census data
* An API implemented using FastAPI that can be used to do model inference

Other features of this repo:
* Data version control using DVC
* Continuous integration using GitHub Actions
* Continuous deployment using Render
* Unit tests for API and ML code using pytest


## GitHub Actions
* Whenever a commit is pushed to main, GitHub Actions will run pytest to ensure all unit tests pass and flake8 to ensure all code follows format guidelines

## Data
* This project uses the UCI ML Repository Census Data dataset which can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)
* EDA can be found in [EDA.ipynb](data/EDA.ipynb), this includes code to clean the data as well as visualize it
* Different versions of the dataset were handled using DVC

## Model
* All ML model code can be found in [ml](ml/), such as a training script, data cleaning and model utility functions. Additionally, model performance on different slices of the test data can be found in [slice_performance.txt](ml/slice_performance.txt)
* Model and encoder files can be found in [model](model/)
* A model card defining model considerations and performance can be found at [model_card.md](model_card.md)

## API Creation
* Code for the API can be found in [main.py](main.py). It implements:
   * GET on the root domain, which provides a welcome message.
   * POST on /predict that does model inference
* API documentation, including examples, can be found at /docs

## API Deployment
* The API can be deployed locally by running `uvicorn main:app`, or it can be continuously deployed using Render by linking this GitHub repo
