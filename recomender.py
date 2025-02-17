import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
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
        
        # Optimización de tipos de datos
        for col in ['main_category', 'sub_category']:
            if col in self.products_df.columns:
                self.products_df[col] = self.products_df[col].astype('category')
        
        self.products_df['combined_features'] = self.products_df.apply(
            lambda x: self.preprocess_text(self.create_product_description(x)), axis=1
        )

        # Generar embeddings con precisión reducida
        self.product_embeddings = self.model.encode(
            self.products_df['combined_features'].tolist(),
            show_progress_bar=True
        ).astype(np.float16)

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
        """Guarda el modelo optimizado en una carpeta específica"""
        os.makedirs(folder_path, exist_ok=True)

        # Guardar DataFrame en formato Parquet comprimido
        self.products_df.to_parquet(
            os.path.join(folder_path, "products.parquet"), 
            compression='gzip'
        )

        # Guardar embeddings en formato binario numpy
        np.save(
            os.path.join(folder_path, "embeddings.npy"), 
            self.product_embeddings,
            allow_pickle=False
        )

        print(f"Modelo optimizado guardado en: {folder_path}")

    @classmethod
    def load_model(cls, folder_path="modelo_recomendador"):
        """Carga un modelo optimizado desde una carpeta"""
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Carpeta no encontrada: {folder_path}")

            instance = cls()
            
            # Cargar DataFrame desde Parquet
            instance.products_df = pd.read_parquet(
                os.path.join(folder_path, "products.parquet")
            )
            
            # Cargar embeddings y convertir a float32 para compatibilidad
            instance.product_embeddings = np.load(
                os.path.join(folder_path, "embeddings.npy")
            ).astype(np.float32)  # Mejor compatibilidad con operaciones posteriores

            print("Modelo cargado exitosamente")
            return instance

        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return None

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

    def update_model(self, new_products_data):
        """Actualiza el modelo con nuevos productos"""
        # Combinar con datos existentes
        updated_df = pd.concat([self.products_df, new_products_data], ignore_index=True)
        
        # Eliminar duplicados manteniendo la última entrada
        updated_df = updated_df.drop_duplicates('product_id', keep='last')
        
        # Reentrenar el modelo con los datos actualizados
        self.fit(updated_df)
        
        return True