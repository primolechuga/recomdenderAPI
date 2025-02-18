from fastapi import FastAPI, Query
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from recomender import ContentBasedRecommender  

app = FastAPI()

# Cargar el modelo de recomendación
recommender = ContentBasedRecommender.load_model(folder_path="./modelo_recomendador")


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
    recomendaciones = recommender.get_recommendations(product_id=product_id, n_recommendations=n_recommendations)
    return recomendaciones.to_dict(orient="records") # Convertir DataFrame a lista de diccionarios

#Se desabilita la función de recomendaciones por texto ya que es muy pesada y no se puede ejecutar en un servidor gratuito
# @app.get("/recommendations/by-text/")
# def get_recommendations_by_text(query: str, n_recommendations: int = Query(5, gt=0)):
#     """
#     Obtiene recomendaciones basadas en un texto de búsqueda.
    
#     - **query**: Texto de búsqueda para el cual se desean recomendaciones.
#     - **n_recommendations**: Número de recomendaciones a devolver (por defecto 5).
#     """
#     recomendaciones = recommender.recommend_from_text(query, n_recommendations=n_recommendations)
#     return recomendaciones.to_dict(orient="records")  # Convertir DataFrame a lista de diccionarios

@app.get("/health")
def health_check():
    return {"status": "ok"}