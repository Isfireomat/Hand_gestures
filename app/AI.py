import tensorflow as tf  # Импорт библиотеки TensorFlow для создания и обучения модели машинного обучения
from add import get_train, split_array  # Импорт функций get_train и split_array из модуля add

# Получение тренировочных данных с помощью функции get_train из указанных файлов
x, y = get_train(["Время.txt", "До свидания.txt", "Здравствуй.txt", "Извиняться.txt", "Помогать.txt", "Спасибо.txt", "Пустота.txt"])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test = split_array(x, 0.8)
y_train, y_test = split_array(y, 0.8)

# Определение архитектуры модели с помощью последовательного API
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(84,)),  # Входной слой с размерностью признаков
    tf.keras.layers.BatchNormalization(),  # Нормализация данных
    tf.keras.layers.Dense(512, activation='relu'),  # Полносвязный слой с 512 нейронами и функцией активации ReLU
    tf.keras.layers.Dropout(0.5),  # Слой Dropout для предотвращения переобучения
    tf.keras.layers.Dense(256, activation='relu'),  # Полносвязный слой с 256 нейронами и функцией активации ReLU
    tf.keras.layers.Dropout(0.5),  # Слой Dropout для предотвращения переобучения
    tf.keras.layers.Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и функцией активации ReLU
    tf.keras.layers.Dropout(0.5),  # Слой Dropout для предотвращения переобучения
    tf.keras.layers.Dense(64, activation='relu'),  # Полносвязный слой с 64 нейронами и функцией активации ReLU
    tf.keras.layers.Dropout(0.5),  # Слой Dropout для предотвращения переобучения
    tf.keras.layers.Dense(7, activation='softmax')  # Выходной слой с функцией активации softmax для классификации на 7 классов
])

# Компиляция модели с использованием оптимизатора 'adam' и функции потерь 'categorical_crossentropy'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на обучающих данных, с валидацией на тестовых данных, в течение 1000 эпох
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')  # Вывод потери и точности на тестовых данных

# Сохранение модели в файл 'gesture_recognition_model.h5'
model.save('gesture_recognition_model.h5')
