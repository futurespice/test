import uvicorn
import sys
import logging
import webbrowser
import threading
import time
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server-runner")

# Проверка наличия необходимых файлов
required_files = ["main.py", "static/index.html"]
for file_path in required_files:
    if not os.path.exists(file_path):
        logger.error(f"Отсутствует необходимый файл: {file_path}")
        print(f"ОШИБКА: Отсутствует необходимый файл: {file_path}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)

# Создание папки для статических файлов, если её нет
if not os.path.exists("static"):
    logger.info("Создание папки static")
    os.makedirs("static")


def open_browser():
    """Функция для открытия браузера с небольшой задержкой"""
    time.sleep(2)
    url = "http://localhost:8000"
    logger.info(f"Открытие браузера по адресу: {url}")
    print(f"Открытие веб-браузера: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    try:
        print("=== Запуск сервера сравнения QR-кодов ЭБД ===")
        print("Для остановки сервера нажмите Ctrl+C\n")

        logger.info("Запуск сервера")

        # Запуск браузера в отдельном потоке
        threading.Thread(target=open_browser).start()

        # Запуск сервера
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
        print("\nСервер остановлен. Спасибо за использование!")
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}")
        print(f"\nПроизошла ошибка: {str(e)}")
        input("Нажмите Enter для выхода...")