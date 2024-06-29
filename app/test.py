import cv2  # Импорт библиотеки OpenCV для работы с изображениями и видео
import mediapipe as mp  # Импорт библиотеки Mediapipe для работы с различными алгоритмами машинного обучения
import numpy as np  # Импорт библиотеки numpy для работы с массивами
import tensorflow as tf  # Импорт библиотеки TensorFlow для работы с нейронными сетями

# Инициализация видеозахвата с камеры (0 - индекс камеры)
cap = cv2.VideoCapture(0)

# Инициализация объекта для обнаружения ладоней с максимальным числом обнаруженных ладоней равным 2
hands = mp.solutions.hands.Hands(max_num_hands=2)

# Инициализация объекта для рисования на изображении
draw = mp.solutions.drawing_utils

# Переменная для управления записью данных в файл
bool = False

# Счетчик кадров
frame = 0

# Открытие файла для записи данных
f = open('Пустота.txt', 'a')

# Начальный размер для записи данных
size = 10000

# Бесконечный цикл для обработки видеопотока
while True:
    # Закрытие окна при нажатии клавиши Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    # Включение/выключение записи данных при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if bool: 
            bool = False
            print("stop")
        else: 
            bool = True
            print("start")
            
    # Считывание кадра с камеры
    success, image = cap.read()
    
    # Отражение изображения по вертикали для корректного отображения
    image = cv2.flip(image, -1)
    
    # Преобразование изображения в цветовое пространство RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обработка изображения алгоритмом обнаружения ладоней
    results = hands.process(imageRGB)
    
    # Проверка наличия обнаруженных ключевых точек ладоней
    if results.multi_hand_landmarks:
        keypoints_array = np.zeros(84)
        for hand_landmarks in results.multi_hand_landmarks:
            # Преобразование координат ключевых точек в одномерный массив
            hand_keypoints = [hand_landmarks.landmark[i].x-hand_landmarks.landmark[0].x for i in range(21)]
            hand_keypoints += [hand_landmarks.landmark[i].y-hand_landmarks.landmark[0].y for i in range(21)]
            
            # Определение правой или левой руки и запись данных в соответствующую часть массива
            if results.multi_handedness:
                for hand_handedness in results.multi_handedness:
                    if hand_handedness.classification[0].label == "Right": 
                        keypoints_array[:42] = hand_keypoints
                    elif hand_handedness.classification[0].label == "Left": 
                        keypoints_array[42:] = hand_keypoints
        
        # Запись данных в файл при активированном флаге bool
        if bool:
            print(f"size:{size}")
            f.write('[{}]\n'.format(','.join(map(str, keypoints_array))))
            size -= 1
            if size == 0: 
                bool = False
            
        # Рисование ладоней на изображении
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)
    
    # Отображение обработанного изображения в окне
    cv2.imshow("Hand", image)

# Закрытие видеозахвата и окон при завершении работы программы
cap.release()
cv2.destroyAllWindows()
