import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

class ContentBasedRecommender:
    def __init__(self):
        self.product_embeddings = None
        self.products_df = None

    @classmethod
    def load_model(cls, folder_path="modelo_recomendador"):
        """Carga un modelo previamente guardado desde una carpeta"""
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Carpeta no encontrada: {folder_path}")

            instance = cls()
            
            # Cargar DataFrame desde Parquet
            instance.products_df = pd.read_parquet(
                os.path.join(folder_path, "products.parquet")
            )
            
            # Cargar embeddings
            instance.product_embeddings = np.load(
                os.path.join(folder_path, "embeddings.npy")
            )

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