import os  # Импорт модуля os для взаимодействия с операционной системой

# Вызов команды в командной строке для преобразования файла интерфейса пользователя в формате PyQt6 (.ui) в файл Python (.py),
# который содержит код, созданный из этого интерфейса.
os.system("pyuic6 untitled.ui -o main_form.py")