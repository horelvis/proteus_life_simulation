FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Instalar el paquete
RUN pip install -e .

# Exponer puerto para posibles visualizaciones web
EXPOSE 8080

# Comando por defecto
CMD ["python", "main.py", "demo"]