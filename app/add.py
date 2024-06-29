import numpy as np  # Импорт библиотеки numpy с псевдонимом np для работы с массивами
import random  # Импорт модуля random для работы с случайными числами

def split_array(array, size):  # Определение функции split_array для разделения массива
    split_index = int(len(array) * size)  # Вычисление индекса для разделения массива
    array_size_percent = array[:split_index]  # Получение процентной части массива
    array_size_percent = array[split_index:]  # Получение оставшейся части массива   
    return array_size_percent, array_size_percent  # Возврат разделенных массивов

def shuffle_arrays(array1, array2):  # Определение функции shuffle_arrays для перемешивания массивов
    indices = list(range(len(array1)))  # Создание списка индексов
    random.shuffle(indices)  # Перемешивание индексов
    array1_shuffled = [array1[i] for i in indices]  # Перемешивание первого массива
    array2_shuffled = [array2[i] for i in indices]  # Перемешивание второго массива    
    return array1_shuffled, array2_shuffled  # Возврат перемешанных массивов

def get_train(masiv):  # Определение функции get_train для получения тренировочных данных
    X_train = []  # Инициализация списка для хранения признаков
    Y_train = []  # Инициализация списка для хранения меток
    size = len(masiv)  # Получение размера массива
    for i, v in enumerate(masiv):  # Цикл по индексам и элементам массива
        arr = [0] * size  # Создание массива для метки
        arr[i] = 1  # Установка соответствующего значения метки
        with open(v, "r") as f:  # Открытие файла для чтения
            for l in f:  # Чтение строк из файла
                X_train.append(eval(l))  # Добавление признаков в список
                Y_train.append(arr)  # Добавление меток в список
    X_train, Y_train = shuffle_arrays(X_train, Y_train)  # Перемешивание тренировочных данных
    print(Y_train)  # Вывод меток в консоль (для отладки)
    return np.array(X_train), np.array(Y_train)  # Возврат тренировочных данных в виде массивов numpy

if __name__ == "__main__":  # Проверка, что скрипт запущен как основная программа
    get_train(["Время.txt", "Пусто.txt"])  # Вызов функции get_train с указанием файлов данных
