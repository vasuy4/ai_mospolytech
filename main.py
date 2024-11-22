import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from PIL import Image
while True:
    # Путь к датасету
    dataset_path = 'mospolytech_dataset'

    # Функция для загрузки изображений
    def load_images_from_folder(folder):
        images = []
        labels = []
        class_names = os.listdir(folder)
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(folder, class_name)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path).resize((32, 32))  # Изменяем размер изображений
                img = np.array(img) / 255.0  # Нормализация данных
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels), class_names

    # Загрузка изображений и меток
    images, labels, class_names = load_images_from_folder(dataset_path)

    # Создание датасета
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Перемешивание и разбиение на обучающий и тестовый наборы
    dataset = dataset.shuffle(len(images))
    train_size = int(0.8 * len(images))
    train_dataset = dataset.take(train_size).batch(32)
    test_dataset = dataset.skip(train_size).batch(32)

    # Создание архитектуры сети
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names))
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Обучение нейронной сети
    history = model.fit(train_dataset, epochs=50, validation_data=test_dataset)

    # Анализ результатов
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='Точность обучения')
    plt.plot(history.history['val_accuracy'], label='Точность валидации')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Функция для предсказания класса изображения
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
    print(f"Total: {total}, correctly: {correctly}. The percentage of guessing: {correctly/total * 100}")
    if correctly/total * 100 >= 99.99:

        model.save('mospolytech_model.keras')


        break