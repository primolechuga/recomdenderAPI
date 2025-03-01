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
df = pd.read_csv("df_clean.csv")

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
    
@app.get("/getRandomProducts")
def getRandomProducts():
    """
    Devuelve aleatoriamente 5 productos en formato JSON.
    """
    try:
        # Seleccionar aleatoriamente 5 productos
        random_products = df.sample(n=5).to_dict(orient="index")
        
        return random_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar los productos: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}