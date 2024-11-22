import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from PIL import Image

model_path = "../MosPolyScheduleBot-main/models/predict_image_models/keras_files/mospolytech_model.keras"

class_names = ['Автозаводская_16', 'Б_Семёновская_38', 'Прянишникова_2а']
model = models.load_model(model_path)


def predict_image(model, image_path):
    # Загрузка и предобработка изображения
    image = Image.open(image_path).resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Добавляем размерность батча

    # Предсказание класса
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)

    # Вывод результата
    return class_names[predicted_class[0]]


print("\n\n==TEST==\n")
class_to_name = {
    "Прянишникова_2а": "kop",
    "Б_Семёновская_38": "bs",
    "Автозаводская_16": "avtaz"
}
# Пример использования
# Тестирование:
total = 0
correctly = 0
for img in os.listdir("test"):
    total += 1
    image_path = os.path.join("test", img)
    print("---------------------------")
    print(image_path)
    build_predict = predict_image(model, image_path)
    print(f"Изображение классифицировано как: {build_predict}")
    if class_to_name[build_predict] in img:
        correctly += 1
print(f"Total: {total}, correctly: {correctly}. The percentage of guessing: {correctly / total * 100}")