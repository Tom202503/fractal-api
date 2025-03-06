
# Використовуємо Python 3.9
FROM python:3.9

# Встановлюємо залежності
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код
COPY . .

# Запускаємо сервер
CMD ["uvicorn", "fractal_api:app", "--host", "0.0.0.0", "--port", "8000"]
