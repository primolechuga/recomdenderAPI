import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import pickle  # Importar pickle para manejar archivos .pkl
from joblib import load

class ContentBasedRecommender:
    def __init__(self):
        self.product_embeddings = None
        self.products_df = None

    @classmethod
    def load_model(cls, folder_path="modelo_recomendador"):
        """
        Carga un modelo previamente guardado desde una carpeta usando archivos .pkl.
        
        Args:
            folder_path (str): Ruta a la carpeta del modelo
            
        Returns:
            ContentBasedRecommender: Instancia del recomendador o None si hay error
        """
        try:
            model_path = Path(folder_path).resolve()
            print(f"Intentando cargar modelo desde: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Carpeta no encontrada: {model_path}")
            
            products_file = model_path / "products.pkl"  # Archivo .pkl para el DataFrame
            embeddings_file = model_path / "embeddings.pkl"  # Archivo .pkl para los embeddings
            
            print(f"Buscando archivo products.pkl en: {products_file}")
            print(f"Buscando archivo embeddings.pkl en: {embeddings_file}")
            
            if not products_file.exists():
                raise FileNotFoundError(f"Archivo products.pkl no encontrado en {model_path}")
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Archivo embeddings.pkl no encontrado en {model_path}")
            
            instance = cls()
            
            # Cargar DataFrame desde .pkl
            try:
                instance.products_df = load(products_file)
                print(f"Archivo products.pkl cargado exitosamente. Shape: {instance.products_df.shape}")
            except Exception as e:
                raise Exception(f"Error al cargar products.pkl: {str(e)}")
            
            # Cargar embeddings desde .pkl
            try:
                instance.product_embeddings = load(embeddings_file)
                print(f"Archivo de embeddings.pkl cargado exitosamente. Shape: {instance.product_embeddings.shape}")
            except Exception as e:
                raise Exception(f"Error al cargar embeddings.pkl: {str(e)}")
            
            print("Modelo cargado exitosamente")
            return instance

        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return None
        
    def get_recommendations(self, product_id, n_recommendations=5):
        """Obtiene recomendaciones basadas en un ID de producto"""
        if product_id not in self.products_df['product_id'].values:
            raise ValueError("ID de producto no encontrado")

        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        similarity_scores = cosine_similarity(
            [self.product_embeddings[product_idx]],
            self.product_embeddings
        )[0]

        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        recommendations = self.products_df.iloc[similar_indices][
            ['product_id', 'name', 'main_category', 'sub_category', 'ratings']
        ].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        return recommendations

    def get_product_details(self, product_id):
        """Obtiene detalles de un producto específico"""
        if product_id not in self.products_df['product_id'].values:
            return None
        
        return self.products_df[self.products_df['product_id'] == product_id].iloc[0].to_dict()

    def batch_recommendations(self, product_ids, n_recommendations=5):
        """Genera recomendaciones para múltiples productos a la vez"""
        all_recommendations = {}
        for pid in product_ids:
            try:
                all_recommendations[pid] = self.get_recommendations(pid, n_recommendations)
            except ValueError:
                continue
        return all_recommendations