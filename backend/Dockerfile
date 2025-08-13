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

# Copy requirements files
COPY requirements*.txt ./

# Install PyTorch with CUDA support first (separate to avoid conflicts)
RUN pip install --no-cache-dir -r requirements-torch.txt

# Then install other requirements (without PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all modules and scripts
COPY proteus /app/proteus
COPY app /app/app
COPY arc /app/arc
COPY *.py /app/
COPY entrypoint.sh /app/entrypoint.sh

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Use virtual display for headless OpenGL
ENV DISPLAY=:99

# Expose ports for both servers
EXPOSE 8000 8765

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Run with virtual display
CMD ["/app/entrypoint.sh"]