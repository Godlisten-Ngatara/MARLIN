FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user and give it ownership of the application directory
RUN useradd -m appuser && chown -R appuser /app

# Switch to the non-root user for running the application
USER appuser

EXPOSE 8080

CMD ["uvicorn", "marlin_dhis2:app", "--host", "0.0.0.0", "--port", "8080"]


