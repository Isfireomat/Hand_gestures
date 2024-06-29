# import transmutation  # Импорт пользовательского модуля "transmutation"
import sys  # Импорт модуля sys для системных параметров и функций
import cv2  # Импорт библиотеки OpenCV для обработки изображений
import mediapipe as mp  # Импорт библиотеки Mediapipe для различных решений ИИ
import numpy as np  # Импорт библиотеки numpy для числовых операций
import tensorflow as tf  # Импорт библиотеки TensorFlow для задач машинного обучения
from PyQt6.QtWidgets import QApplication, QMainWindow  # Импорт необходимых классов из библиотеки PyQt6 для разработки GUI
from PyQt6.QtCore import QTimer  # Импорт класса QTimer из PyQt6 для запуска повторяющихся действий
from PyQt6.QtGui import QImage, QPixmap  # Импорт необходимых классов для отображения изображений в PyQt6
from main_form import Ui_MainWindow  # Импорт класса Ui_MainWindow из модуля main_form
from voice import start  # Импорт функции start из модуля voice

class main(QMainWindow, Ui_MainWindow):  # Определение класса main, унаследованного от QMainWindow и Ui_MainWindow
    def __init__(self, parent=None):  # Конструктор для инициализации класса
        super(main, self).__init__(parent)  # Вызов конструктора родительских классов
        self.setupUi(self)  # Настройка пользовательского интерфейса, определенного в Ui_MainWindow
        for i in range(self.get_num_video_channels()):  # Цикл по доступным видеоканалам
            self.comboBox.addItem(str(i))  # Добавление каждого номера канала в элемент comboBox
        self.pushButton.clicked.connect(self.start_video)  # Подключение метода start_video к сигналу clicked кнопки pushButton
        self.pushButton_2.clicked.connect(self.stop_video)  # Подключение метода stop_video к сигналу clicked кнопки pushButton_2
        self.pushButton_2.setEnabled(False)  # Начальное отключение кнопки pushButton_2
        self.timer = QTimer(self)  # Создание объекта QTimer
        self.timer.timeout.connect(self.update_frame)  # Подключение сигнала timeout таймера к методу update_frame
        self.is_playing = False  # Переменная для отслеживания воспроизведения видео
        self.video = cv2.VideoCapture(0)  # Инициализация захвата видео с камеры по умолчанию
        self.model = tf.keras.models.load_model('gesture_recognition_model.h5')  # Загрузка предварительно обученной модели распознавания жестов
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)  # Инициализация модуля рук из Mediapipe
        self.draw = mp.solutions.drawing_utils  # Инициализация модуля утилит рисования из Mediapipe
        self.text_queue = start()  # Запуск функции start из модуля voice

    def start_video(self):  # Метод для запуска захвата видео
        self.frame = 0  # Инициализация счетчика кадров
        self.video = cv2.VideoCapture(int(self.comboBox.currentText()))  # Начало захвата видео с выбранного канала
        self.is_playing = True  # Установка флага, указывающего на воспроизведение видео
        self.pushButton.setEnabled(False)  # Отключение кнопки start
        self.pushButton_2.setEnabled(True)  # Включение кнопки stop
        self.timer.start(33)  # Запуск таймера с интервалом 33 миллисекунды

    def stop_video(self):  # Метод для остановки захвата видео
        self.is_playing = False  # Установка флага, указывающего на остановку воспроизведения видео
        self.pushButton.setEnabled(True)  # Включение кнопки start
        self.pushButton_2.setEnabled(False)  # Отключение кнопки stop
        self.timer.stop()  # Остановка таймера

    def update_frame(self):  # Метод для обновления отображаемого видеокадра
        success, image = self.video.read()  # Чтение кадра из видеозахвата
        if success:  # Проверка успешности чтения кадра
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование кадра в формат RGB
            results = self.hands.process(imageRGB)  # Обработка кадра для обнаружения ключевых точек рук
            if results.multi_hand_landmarks:  # Проверка обнаружения ключевых точек рук
                keypoints_array = np.zeros(84)  # Инициализация массива для хранения ключевых точек рук
                for hand_landmarks in results.multi_hand_landmarks:  # Цикл по обнаруженным ключевым точкам рук
                    hand_keypoints = [hand_landmarks.landmark[i].x - hand_landmarks.landmark[0].x for i in range(21)]  # Вычисление относительных координат x ключевых точек
                    hand_keypoints += [hand_landmarks.landmark[i].y - hand_landmarks.landmark[0].y for i in range(21)]  # Вычисление относительных координат y ключевых точек
                    if results.multi_handedness:  # Проверка обнаружения стороны руки
                        for hand_handedness in results.multi_handedness:  # Цикл по обнаруженным сторонам рук
                            if hand_handedness.classification[0].label == "Right":  # Проверка, что обнаружена правая сторона руки
                                keypoints_array[:42] = hand_keypoints  # Сохранение ключевых точек правой руки
                            elif hand_handedness.classification[0].label == "Left":  # Проверка, что обнаружена левая сторона руки
                                keypoints_array[42:] = hand_keypoints  # Сохранение ключевых точек левой руки
                self.frame += 1  # Увеличение счетчика кадров
                if not self.frame % 72:  # Проверка каждого 72-го кадра
                    predictions = self.model.predict(np.reshape(keypoints_array, (1, 84)))  # Предсказание с использованием модели распознавания жестов
                    text = ["Время", "До свидания", "Здравствуй", "Извиняться", "Помогать", "Спасибо", "Пустота"][np.argmax(predictions[0])]  # Определение предсказанного жеста
                    if text != "Пустота":  # Проверка, что жест не пустой
                        self.label_3.setText(text)  # Отображение предсказанного жеста
                    else:  
                        self.label_3.setText("")  # Очистка метки, если жест пустой
                    if self.checkBox.isChecked() and text != "Пустота":  # Проверка, включен ли голосовой вывод и жест не пустой
                        self.text_queue.put(text)  # Отправка предсказанного жеста на голосовой вывод
                for handLms in results.multi_hand_landmarks:  # Цикл по обнаруженным ключевым точкам рук
                    for id, lm in enumerate(handLms.landmark):  # Цикл по каждой точке ключа в руке
                        h, w, c = image.shape  # Получение высоты, ширины и количества каналов изображения
                        cx, cy = int(lm.x * w), int(lm.y * h)  # Вычисление координат пикселя для точки ключа
                    self.draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)  # Рисование точек ключа и соединений на руке
            h, w, ch = image.shape  # Получение высоты, ширины и количества каналов изображения
            bytes_per_line = ch * w  # Вычисление байтов на строку для изображения
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)  # Создание QImage из данных изображения
            pixmap = QPixmap.fromImage(qt_image)  # Создание QPixmap из QImage
            self.label_2.setPixmap(pixmap)  # Отображение QPixmap на метке

    def get_num_video_channels(self):  # Метод для получения количества доступных видеоканалов
        num_channels = 0  # Инициализация счетчика доступных каналов
        for i in range(10):  # Цикл по диапазону индексов каналов
            cap = cv2.VideoCapture(i)  # Попытка открытия видеозахвата для текущего канала
            if not cap.isOpened():  # Проверка, что захват не открыт
                break  # Выход из цикла, если захват не удалось открыть
            num_channels += 1  # Увеличение счетчика доступных каналов
            cap.release()  # Освобождение объекта видеозахвата
        return num_channels  # Возвращение количества доступных каналов

if __name__ == "__main__":  # Проверка, что скрипт запущен как основная программа
    app = QApplication(sys.argv)  # Создание экземпляра QApplication
    app.setStyle('Fusion') #Устанавливаем стиль
    player = main()  # Создание экземпляра класса main
    player.show()  # Отображение главного окна
    sys.exit(app.exec())  # Выполнение цикла событий приложения и завершение при завершении
