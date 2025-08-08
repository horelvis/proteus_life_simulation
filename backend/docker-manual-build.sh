#!/bin/bash

echo "Manual Docker build process..."

# Step 1: Create a container from base image
echo "1. Creating container from python:3.11..."
docker create --name proteus-build python:3.11

# Step 2: Copy files
echo "2. Copying files..."
docker cp requirements.txt proteus-build:/app/requirements.txt
docker cp proteus proteus-build:/app/proteus
docker cp run_server.py proteus-build:/app/run_server.py
docker cp proteus_vispy.py proteus-build:/app/proteus_vispy.py

# Step 3: Install dependencies
echo "3. Installing dependencies..."
docker start proteus-build
docker exec proteus-build bash -c "cd /app && pip install --no-cache-dir -r requirements.txt"

# Step 4: Commit as new image
echo "4. Creating image..."
docker commit proteus-build proteus-backend:manual

# Step 5: Clean up
echo "5. Cleaning up..."
docker rm -f proteus-build

echo "✅ Done! Run with:"
echo "docker run -p 8000:8000 -w /app proteus-backend:manual python run_server.py"