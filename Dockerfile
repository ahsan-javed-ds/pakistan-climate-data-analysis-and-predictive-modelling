# Using a lightweight Python base image
FROM python:3.9-slim

# Setting 'Working Directory' inside the container
WORKDIR /app

# Copying dependencies list
COPY requirements.txt

# Installation of the dependencies
RUN pip install --no-cache-dir -r -requirements.txt

# Copying the Application Code and Model files
COPY fastapi_app.py
COPY *.joblib .

# Expose the port FastAPI will be running
EXPOSE 8000

# Command To run the application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]