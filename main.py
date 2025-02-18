from fastapi import FastAPI, Query
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from recomender import SistemaRecomendacion  
import tensorflow as tf
import pandas as pd

app = FastAPI()

def cargar_datos(ruta: str = './datos_productos.csv') -> pd.DataFrame:
    """Carga los datos desde disco"""
    print("Cargando datos...")
    return pd.read_csv(ruta, index_col=0)


def cargar_modelo(ruta: str = './modelo_recomendacion.keras') -> tf.keras.Model:
    """Carga el modelo desde disco"""
    print("Cargando modelo...")
    return tf.keras.models.load_model(ruta)

data = cargar_datos()
model = cargar_modelo()
recommender = SistemaRecomendacion(model, data)

# origins = [
#     "http://localhost",
#     "http://localhost:8000",
    
# ]

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
    nombre,recomendaciones = recommender.mostrar_recomendaciones(product_id, top_n= n_recommendations)
    return {"product_name": nombre, "recommendations": recomendaciones}

@app.get("/health")
def health_check():
    return {"status": "ok"}