FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the package
RUN pip install -e .

# Create a non-root user to run the application
RUN groupadd -g 1000 gemini && \
    useradd -u 1000 -g gemini -s /bin/bash -m gemini && \
    chown -R gemini:gemini /app

USER gemini

# Set the entrypoint to the CLI script
ENTRYPOINT ["gemini-update"]

# Default command (show help)
CMD ["--help"]
