
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Завантажуємо модель
try:
    model = joblib.load("fractal_logistic_model_final.joblib")
except Exception as e:
    model = None
    print(f"⚠️ Помилка завантаження моделі: {e}")

# Ініціалізуємо API
app = FastAPI()

# Формат вхідних даних
class FractalInput(BaseModel):
    text: str

# Функція класифікації
@app.post("/classify/")
def classify_fractal(fractal: FractalInput):
    if model is None:
        return {"error": "Модель не завантажена. Переконайтеся, що файл fractal_logistic_model_final.joblib існує."}

    # Спрощене перетворення тексту у вектор (потрібно підключити справжній векторизатор)
    vector = np.array([[len(fractal.text)]])  
    prediction = model.predict(vector)[0]

    return {"category": "code" if prediction == 1 else "text"}

# Головна сторінка
@app.get("/")
def read_root():
    return {"message": "Fractal Classification API is running!"}
