# app/Dockerfile

# Используйте официальный образ Python
FROM python:3.9-slim

# Установите переменную окружения для установки кодировки UTF-8
ENV PYTHONUNBUFFERED=1

# Установите рабочую директорию в /app
WORKDIR /app

# Установите зависимости, которые могут потребоваться для вашего приложения
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Скопируйте все файлы вашего приложения в текущую директорию в контейнере
COPY . .

# Установите зависимости проекта
RUN pip3 install -r requirements.txt

# Установите порт, который будет использоваться вашим приложением Streamlit
EXPOSE 8501

# Запустите ваше приложение Streamlit при запуске контейнера
CMD ["streamlit", "run", "person_identity.py", "--server.port=8501", "--server.address=0.0.0.0"]