FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port (Hugging Face Spaces expects apps to run on port 7860 by default)
EXPOSE 7860

# Run the application using gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
