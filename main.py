from fastapi import FastAPI, Query, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from recomender import ContentBasedRecommender
import tensorflow as tf
import pandas as pd
import requests
import zipfile
import io
import os

app = FastAPI()


recommender = ContentBasedRecommender(model_path="modelo_recomendador")
recommender.load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Bienvenido al sistema de recomendación"}

@app.get("/recommendations/by-product-id/")
def get_recommendations_by_product_id(product_id: int, n_recommendations: int = Query(5, gt=0)):
    """
    Obtiene recomendaciones basadas en un product_id.
    
    - **product_id**: ID del producto para el cual se desean recomendaciones.
    - **n_recommendations**: Número de recomendaciones a devolver (por defecto 5).
    """
    recomendaciones = recommender.get_recommendations(product_id, n_recommendations).to_dict(orient="records")
    return recomendaciones
    

@app.get("/health")
def health_check():
    return {"status": "ok"}