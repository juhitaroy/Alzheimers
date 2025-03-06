# Use the official Python image
FROM python:3.10

# Set the working directory to the Alzheimers folder
WORKDIR /app/Alzheimers

# Copy the requirements file
COPY Alzheimers/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Alzheimers folder into the Docker image
COPY Alzheimers/ .

# Expose port 8000
EXPOSE 8000

# Set the command to run the FastAPI app
CMD ["uvicorn", "server_alz:app", "--host", "0.0.0.0", "--port", "8000"]

