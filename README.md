# White Wine Quality Prediction

Individual Assessment for Software Engineering Practices in Machine Learning (SEPL), 2026

## Project Overview

This project implements an asynchronous machine learning inference
system for predicting white wine quality based on physicochemical
measurements.

## Dataset

UCI Wine Quality Dataset (White Wine)
- 4,898 observations
- 11 input features
- Binary classification:
  - Good quality: score >= 6
  - Lower quality: score < 6

## Technologies

- Python 3.12
- scikit-learn
- FastAPI
- Redis
- Docker and Docker Compose
- pytest and Ruff
- GitHub Actions
- Slurm

## Architecture

Client -> FastAPI -> Redis Queue -> ML Worker -> Redis Results

## Status

Project setup and dataset preparation.