import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os

class ContentBasedRecommender:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_embeddings = None
        self.products_df = None

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return text
        return ''

    def create_product_description(self, row):
        return f"{row['name']} {row['main_category']} {row['sub_category']}"

    def fit(self, products_data):
        if 'product_id' not in products_data.columns:
            products_data['product_id'] = range(len(products_data))

        self.products_df = products_data.copy()
        self.products_df['combined_features'] = self.products_df.apply(
            lambda x: self.preprocess_text(self.create_product_description(x)), axis=1
        )

        self.product_embeddings = self.model.encode(
            self.products_df['combined_features'].tolist(),
            show_progress_bar=True
        )
        return self

    def get_recommendations(self, product_id, n_recommendations=5):
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

    def recommend_from_text(self, query_text, n_recommendations=5):
        processed_query = self.preprocess_text(query_text)
        query_embedding = self.model.encode([processed_query])[0]

        similarity_scores = cosine_similarity(
            [query_embedding],
            self.product_embeddings
        )[0]

        similar_indices = similarity_scores.argsort()[::-1][:n_recommendations]
        recommendations = self.products_df.iloc[similar_indices][
            ['product_id', 'name', 'main_category', 'sub_category', 'ratings']
        ].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        return recommendations

    def save_model(self, folder_path="modelo_recomendador"):
        """Guarda el modelo y sus datos en una carpeta específica"""
        # Crear la carpeta si no existe
        os.makedirs(folder_path, exist_ok=True)

        # Guardar el DataFrame de productos
        self.products_df.to_pickle(os.path.join(folder_path, "products.pkl"))

        # Guardar los embeddings de productos
        with open(os.path.join(folder_path, "embeddings.pkl"), 'wb') as f:
            pickle.dump(self.product_embeddings, f)

        print(f"Modelo guardado en: {folder_path}")

    @classmethod
    def load_model(cls, folder_path="modelo_recomendador"):
        """Carga un modelo guardado desde una carpeta específica"""
        try:
            # Verificar que existan los archivos necesarios
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"No se encontró la carpeta: {folder_path}")

            products_path = os.path.join(folder_path, "products.pkl")
            embeddings_path = os.path.join(folder_path, "embeddings.pkl")

            if not os.path.exists(products_path) or not os.path.exists(embeddings_path):
                raise FileNotFoundError("No se encontraron los archivos necesarios del modelo")

            # Crear una nueva instancia
            instance = cls()

            # Cargar el DataFrame de productos
            instance.products_df = pd.read_pickle(products_path)

            # Cargar los embeddings de productos
            with open(embeddings_path, 'rb') as f:
                instance.product_embeddings = pickle.load(f)

            print("Modelo cargado exitosamente")
            return instance

        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return None