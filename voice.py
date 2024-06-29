import pyttsx3  # Импорт библиотеки pyttsx3 для синтеза речи
from multiprocessing import Process, Queue  # Импорт классов Process и Queue из библиотеки multiprocessing

def speak(queue):  # Определение функции speak с аргументом queue
    engine = pyttsx3.init()  # Инициализация движка для синтеза речи
    while True:  # Бесконечный цикл
        text = queue.get()  # Получение текста из очереди
        engine.say(text)  # Озвучивание текста
        engine.runAndWait()  # Ожидание завершения озвучивания

def start():  # Определение функции start
    text_queue = Queue()  # Создание новой очереди
    speak_process = Process(target=speak, args=(text_queue,))  # Создание процесса для функции speak с передачей ей очереди в качестве аргумента
    speak_process.start()  # Запуск процесса озвучивания
    return text_queue  # Возврат созданной очереди для добавления текста
