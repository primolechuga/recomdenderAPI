import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union

class SistemaRecomendacion:
    def __init__(self, modelo, dataframe):
        """
        Sistema de recomendación basado en embeddings y similitud coseno
        
        Args:
            modelo (keras.Model): Modelo entrenado con capas de embedding
            dataframe (pd.DataFrame): DataFrame con los datos de productos
        """
        self.modelo = modelo
        self.df = dataframe
        self.mapa_nombres = self._crear_mapa_nombres()
        self.embeddings = self._cargar_embeddings()
        self.metadatos = self._preprocesar_metadatos()
        self.vectores_combinados = self._precalcular_vectores()

    def _crear_mapa_nombres(self) -> Dict[int, str]:
        """Crea diccionario de mapeo ID -> Nombre de producto"""
        return self.df.set_index('name_encoded')['name'].astype(str).to_dict()

    def _cargar_embeddings(self) -> Dict[str, np.ndarray]:
        """Extrae y verifica los embeddings del modelo"""
        try:
            return {
                'producto': self.modelo.get_layer("product_embed").get_weights()[0],
                'categoria_principal': self.modelo.get_layer("main_category_embed").get_weights()[0],
                'subcategoria': self.modelo.get_layer("sub_category_embed").get_weights()[0]
            }
        except ValueError as e:
            raise RuntimeError("El modelo no contiene las capas de embedding requeridas") from e

    def _preprocesar_metadatos(self) -> Dict[int, Dict[str, int]]:
        """Preprocesa todos los metadatos para acceso rápido"""
        return {
            row['name_encoded']: {
                'categoria_principal': row['main_category_encoded'],
                'subcategoria': row['sub_category_encoded']
            }
            for _, row in self.df.iterrows()
        }

    def _precalcular_vectores(self) -> np.ndarray:
        """Calcula eficientemente todos los vectores combinados"""
        # Obtener todos los IDs únicos y ordenados
        ids_productos = sorted(self.metadatos.keys())
        
        # Obtener índices para cada categoría
        main_cat_indices = [self.metadatos[pid]['categoria_principal'] for pid in ids_productos]
        sub_cat_indices = [self.metadatos[pid]['subcategoria'] for pid in ids_productos]
        
        # Crear matriz combinada usando numpy avanzado
        return np.hstack([
            self.embeddings['producto'][ids_productos],
            self.embeddings['categoria_principal'][main_cat_indices],
            self.embeddings['subcategoria'][sub_cat_indices]
        ])

    def _obtener_indice(self, producto_id: int) -> int:
        """Obtiene el índice correspondiente al ID del producto"""
        try:
            return list(self.metadatos.keys()).index(producto_id)
        except ValueError:
            raise KeyError(f"Producto ID {producto_id} no encontrado")

    def generar_recomendaciones(self, producto_id: int, top_n: int = 5) -> List[str]:
        """
        Genera recomendaciones basadas en similitud combinada
        
        Args:
            producto_id (int): ID del producto de referencia
            top_n (int): Número de recomendaciones a devolver
            
        Returns:
            List[str]: Lista de nombres de productos recomendados
        """
        idx = self._obtener_indice(producto_id)
        similitudes = cosine_similarity(
            [self.vectores_combinados[idx]],
            self.vectores_combinados
        )[0]
        
        # Ordenar y excluir el producto original
        indices = np.argsort(-similitudes)[1:top_n+1]
        return [self.mapa_nombres[self.df.iloc[i]['name_encoded']] for i in indices]

    def mostrar_recomendaciones(self, producto_id: int, top_n: int = 5):
        """Muestra las recomendaciones formateadas"""
        try:
            nombre = self.mapa_nombres[producto_id]
            recomendados = self.generar_recomendaciones(producto_id, top_n)

            return nombre,recomendados
                
        except KeyError:
            return "Error: Producto con ID {producto_id} no existe"
        