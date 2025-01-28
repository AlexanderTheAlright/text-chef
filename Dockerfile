# Start from the official Python 3.9-slim image
FROM python:3.9-slim

# Install required system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip via Python so we ensure we're calling the right interpreter
RUN python -m pip install --upgrade pip

# Create and switch to the app directory
WORKDIR /app

# Copy your requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install project dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your code (including textualizer.py) into the container
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "textualizer.py", "--server.port=8501", "--server.address=0.0.0.0"]
