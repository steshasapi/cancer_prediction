# Используем официальный Python 3.8 как базовый образ
FROM python:3.8

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . /app/

# Указываем команду для запуска приложения
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8501"]

