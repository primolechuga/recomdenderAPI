import os
import pickle
import pandas as pd
import re
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, model_path="modelo_recomendador"):
        """
        Inicializa el recomendador.
        :param model_path: Ruta de la carpeta donde se guardan los archivos del modelo.
        """
        self.model_path = model_path
        self.model = None  # Modelo de SentenceTransformer
        self.products_df = None  # DataFrame de productos
        self.product_embeddings = None  # Embeddings de los productos

    def load_model(self):
        """
        Carga el modelo de SentenceTransformer y los embeddings de productos.
        """

        if self.products_df is None or self.product_embeddings is None:
            try:
                # Cargar el DataFrame de productos
                products_path = os.path.join(self.model_path, "products.pkl")
                if os.path.exists(products_path):
                    self.products_df = pd.read_pickle(products_path)

                # Cargar los embeddings de productos
                embeddings_path = os.path.join(self.model_path, "embeddings.pkl")
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, 'rb') as f:
                        self.product_embeddings = pickle.load(f)

                print("Modelo y datos cargados exitosamente.")
            except Exception as e:
                print(f"Error al cargar el modelo o los datos: {str(e)}")

    def preprocess_text(self, text):
        """
        Preprocesa el texto para eliminar caracteres especiales y convertirlo a minúsculas.
        :param text: Texto a preprocesar.
        :return: Texto preprocesado.
        """
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return text
        return ''

    def get_recommendations(self, product_id, n_recommendations=5):
        """
        Obtiene recomendaciones basadas en el ID de un producto.
        :param product_id: ID del producto.
        :param n_recommendations: Número de recomendaciones a devolver.
        :return: DataFrame con las recomendaciones.
        """
        self.load_model()  # Asegurarse de que el modelo y los datos estén cargados

        if product_id not in self.products_df['product_id'].values:
            raise ValueError("ID de producto no encontrado")

        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        similarity_scores = cosine_similarity(
            [self.product_embeddings[product_idx]],
            self.product_embeddings
        )[0]

        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations + 1]
        recommendations = self.products_df.iloc[similar_indices][
            ['product_id', 'name', 'main_category', 'sub_category', 'ratings']
        ].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        return recommendations

    # def recommend_from_text(self, query_text, n_recommendations=5):
    #     """
    #     Obtiene recomendaciones basadas en un texto de búsqueda.
    #     :param query_text: Texto de búsqueda.
    #     :param n_recommendations: Número de recomendaciones a devolver.
    #     :return: DataFrame con las recomendaciones.
    #     """
    #     self.load_model()  # Asegurarse de que el modelo y los datos estén cargados

    #     processed_query = self.preprocess_text(query_text)
    #     query_embedding = self.model.encode([processed_query])[0]

    #     similarity_scores = cosine_similarity(
    #         [query_embedding],
    #         self.product_embeddings
    #     )[0]

    #     similar_indices = similarity_scores.argsort()[::-1][:n_recommendations]
    #     recommendations = self.products_df.iloc[similar_indices][
    #         ['product_id', 'name', 'main_category', 'sub_category', 'ratings']
    #     ].copy()
    #     recommendations['similarity_score'] = similarity_scores[similar_indices]
    #     return recommendations

    def save_model(self, folder_path="modelo_recomendador"):
        """
        Guarda el modelo y sus datos en una carpeta específica.
        :param folder_path: Ruta de la carpeta donde se guardarán los archivos.
        """
        os.makedirs(folder_path, exist_ok=True)

        # Guardar el DataFrame de productos
        self.products_df.to_pickle(os.path.join(folder_path, "products.pkl"))

        # Guardar los embeddings de productos
        with open(os.path.join(folder_path, "embeddings.pkl"), 'wb') as f:
            pickle.dump(self.product_embeddings, f)

        print(f"Modelo guardado en: {folder_path}")
#Ejemplo de uso
#recommender = ContentBasedRecommender.load_model(folder_path="model_Recomender")
# recomendaciones = recommender.get_recommendations(product_id=200, n_recommendations=5)
# print("Recomendaciones por ID:")
# print(recomendaciones)

# # Por texto
# recomendaciones = recommender.recommend_from_text("Running shoes", n_recommendations=5)
# print("\nRecomendaciones por texto:")
# print(recomendaciones)