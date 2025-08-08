# PROTEUS Backend - GPU-Accelerated Container
FROM python:3.11-slim

# Install system dependencies for OpenGL and Numba
RUN apt-get update && apt-get install -y \
    # OpenGL dependencies for Vispy
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    # X11 for GUI (headless mode for server)
    xvfb \
    # Build tools for Numba
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the proteus module and scripts
COPY proteus /app/proteus
COPY *.py /app/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Use virtual display for headless OpenGL
ENV DISPLAY=:99

# Expose port for WebSocket/API
EXPOSE 8000

# Create entrypoint script for xvfb
RUN echo '#!/bin/bash\nXvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\nsleep 2\npython run_server.py' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Run with virtual display
CMD ["/app/entrypoint.sh"]