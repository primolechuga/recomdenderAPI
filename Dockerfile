FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Crear el directorio del modelo
RUN mkdir -p /app/modelo_recomendador

# Copiar los archivos de la aplicación
COPY . .

# Asegurar permisos correctos
RUN chmod -R 755 /app/modelo_recomendador

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]