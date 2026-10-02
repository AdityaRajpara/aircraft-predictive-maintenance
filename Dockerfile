FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

# Data and trained models are not baked into the image. Mount them:
#   docker run -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" apm
# then run `python -m src.predict` once (docker exec) if predictions.duckdb is missing.
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
