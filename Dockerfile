# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OpenCV and FFmpeg video processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for efficient Docker caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . /app

# Expose port 7860 (default port for Hugging Face Spaces Docker SDK)
EXPOSE 7860

# Command to launch Streamlit on Hugging Face Spaces port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]