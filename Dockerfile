FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source and bundled model
COPY app/ ./app/
COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 8000

CMD ["./startup.sh"]
