from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Загрузка модели и масштабатора
model = load_model('neural_model.keras')
scaler = joblib.load('scaler.pkl')

# Параметры модели (в порядке их использования в обучении)
FEATURE_NAMES = [
    'Плотность, кг/м3',
    'модуль упругости, ГПа',
    'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2',
    'Температура вспышки, С_2',
    'Поверхностная плотность, г/м2',
    'Модуль упругости при растяжении, ГПа',
    'Прочность при растяжении, МПа',
    'Потребление смолы, г/м2',
    'Угол нашивки',
    'Шаг нашивки',
    'Плотность нашивки'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Сбор данных из формы
        input_data = {}
        errors = []
        
        # Проверка заполненности и корректности данных
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if not value:
                errors.append(feature)
            else:
                try:
                    # Преобразование в число с проверкой для угла
                    num_value = float(value)
                    if feature == 'Угол нашивки' and num_value not in [0, 90]:
                        errors.append("Угол нашивки должен быть 0 или 90")
                    input_data[feature] = num_value
                except ValueError:
                    errors.append(f"Некорректное число: {feature}")
        
        # Обработка ошибок
        if errors:
            return render_template(
                'index.html', 
                error=f"Ошибки в данных: {', '.join(errors)}"
            )
        
        try:
            # Преобразование данных и масштабирование
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
            input_df = input_df.apply(pd.to_numeric)
            # Проверка на NaN после конвертации
            if input_df.isnull().any().any():
                return render_template('index.html', 
                                       error="Ошибка: Некорректные числовые значения")
            scaled_data = scaler.transform(input_df)
            
            # Прогнозирование
            prediction = model.predict(scaled_data)
            result = prediction[0][0]
            
            return render_template('index.html', result=f"{result:.4f}")
        
        except Exception as e:
            return render_template('index.html', error=f"Ошибка прогноза: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)