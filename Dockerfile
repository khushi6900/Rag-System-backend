# ---- Base image ----
FROM python:3.10.18

# ---- Environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies (required for psycopg2) ----
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Copy requirements first (better caching) ----
COPY requirements.txt .

# ---- Install Python dependencies ----
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# ---- Expose port ----
EXPOSE 8000

# ---- Start FastAPI ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]