# This Dockerfile builds a lightweight container that:
#   1. Uses Python 3.10 slim base
#   2. Installs dependencies via Poetry
#   3. Copies source code
#   4. Exposes API on port 8080
#
# Build image:
#   docker build -t predictor:latest .
#
# Check that the image exists:
#   docker images
#
# Run container (maps host port 48035 → container port 8080):
#   docker run -p 48035:8080 --name predictor predictor:latest
#
# Verify container is running:
#   docker ps -a
#
# Stop the container:
#   docker stop predictor
#
# Remove the container:
#   docker rm predictor
#
# Access the API at:
#   http://localhost:48035/docs


FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY predictor ./predictor

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD ["poetry", "run", "api"]