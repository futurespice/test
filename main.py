from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from io import BytesIO
import logging, re, time, os
from typing import Tuple, Set, List, Dict, Optional
import uuid, datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def clean_code(value) -> Optional[str]:
    """Очистка QR-кода или серии от лишних символов"""
    if pd.isna(value) or value is None:
        return None
    # Удаляем пробелы, кавычки и переводы строк
    return str(value).strip().strip('"').strip("'").replace('\n', '').replace('\r', '')


def is_valid_qr(value) -> bool:
    """Проверка валидности QR-кода (должен содержать много цифр)"""
    if not value or pd.isna(value):
        return False
    value_str = str(value).strip()
    # QR-код должен содержать достаточное количество цифр
    digit_count = sum(c.isdigit() for c in value_str)
    return digit_count >= 10 and len(value_str) >= 15


def is_valid_series(value) -> bool:
    """Проверка валидности серии (обычно 4-12 символов)"""
    if not value or pd.isna(value):
        return False
    value_str = str(value).strip()
    # Серийный номер обычно состоит из 4-12 символов
    return len(value_str) >= 4 and len(value_str) <= 20 and sum(c.isdigit() or c.isalpha() for c in value_str) >= 3


def find_qr_series_columns(df):
    """Находит колонки с QR-кодами и сериями в DataFrame"""
    qr_column = None
    series_column = None

    # 1. Сначала ищем по названиям колонок
    for col in df.columns:
        col_str = str(col).lower()

        # Ищем колонку с серийными номерами по имени
        if any(keyword in col_str for keyword in ['серия', 'серии', 'series', 'партия', 'серийный']):
            series_column = col
            logger.info(f"Найдена колонка с серийными номерами по имени: {col}")

        # Ищем колонку с QR-кодами по имени
        if any(keyword in col_str for keyword in ['qr', 'код', 'штрих', 'штрихкод', 'datamatrix']):
            qr_column = col
            logger.info(f"Найдена колонка QR по имени: {col}")

    # 2. Если серийный номер не найден по имени, ищем по содержимому
    if series_column is None:
        series_counts = {}
        for col in df.columns:
            # Проверяем только первые 100 строк для ускорения
            sample = df[col].head(100)
            valid_series = sample.apply(is_valid_series).sum()
            if valid_series > 0:
                series_counts[col] = valid_series

        if series_counts:
            # Выбираем колонку с наибольшим количеством серий
            series_column = max(series_counts.items(), key=lambda x: x[1])[0]
            logger.info(
                f"Найдена колонка с серийными номерами по содержимому: {series_column} ({series_counts[series_column]} серий)")

    # 3. Если QR не найден по имени, ищем по содержимому
    if qr_column is None:
        qr_counts = {}
        for col in df.columns:
            if col != series_column:  # Исключаем колонку с сериями
                # Проверяем только первые 100 строк для ускорения
                sample = df[col].head(100)
                valid_qrs = sample.apply(is_valid_qr).sum()
                if valid_qrs > 0:
                    qr_counts[col] = valid_qrs

        if qr_counts:
            # Выбираем колонку с наибольшим количеством QR-кодов
            qr_column = max(qr_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Найдена колонка QR по содержимому: {qr_column} ({qr_counts[qr_column]} QR-кодов)")

    # 4. Если не нашли серийный номер, используем первую колонку
    if series_column is None and len(df.columns) > 0:
        series_column = df.columns[0]
        logger.info(f"Колонка с серийными номерами не найдена, используем первую колонку: {series_column}")

    # 5. Если все еще не нашли QR, и есть минимум 2 колонки, используем вторую колонку
    if qr_column is None and len(df.columns) > 1:
        qr_column = df.columns[1] if df.columns[0] == series_column else df.columns[0]
        logger.info(f"QR-колонка не найдена, используем колонку: {qr_column}")

    return series_column, qr_column


def safe_read_excel(file_data, header=0):
    """Безопасное чтение Excel-файла с несколькими попытками разных параметров"""
    try:
        # Первая попытка - стандартное чтение
        df = pd.read_excel(BytesIO(file_data), header=header, dtype=str)
        if not df.empty:
            return df

        # Если получили пустой DataFrame, пробуем другие подходы
        # Попытка 2 - без указания заголовка
        df = pd.read_excel(BytesIO(file_data), header=None, dtype=str)
        if not df.empty:
            return df

        # Попытка 3 - с явным указанием движка
        df = pd.read_excel(BytesIO(file_data), header=header, engine='openpyxl', dtype=str)
        if not df.empty:
            return df

        # Попытка 4 - с минимальными параметрами
        df = pd.read_excel(BytesIO(file_data), dtype=str)

        return df
    except Exception as e:
        logger.error(f"Ошибка при чтении Excel: {str(e)}")
        raise ValueError(f"Не удалось прочитать Excel-файл: {str(e)}")


def extract_serial_from_qr(qr_code: str, serial_numbers: Set[str]) -> Optional[str]:
    """Пытается найти серийный номер внутри QR-кода"""
    if not qr_code:
        return None


    for serial in serial_numbers:
        if serial and serial in qr_code:
            return serial

    return None


async def compare_qr_codes(file1_data, file2_data):
    """Сравнение серийных номеров из первого файла с QR-кодами из второго файла"""
    start_time = time.time()
    try:
        logger.info("Начинаем сравнение файлов")

        # Чтение первого файла (с серийными номерами)
        try:
            df1 = safe_read_excel(file1_data)
            logger.info(f"Первый файл успешно прочитан. Размер: {df1.shape}, Колонки: {df1.columns.tolist()}")

            if df1.empty:
                raise HTTPException(status_code=400, detail="Первый файл пуст.")

            # Выводим первые несколько строк для отладки
            logger.info(f"Пример данных из первого файла:\n{df1.head().to_string()}")
        except Exception as e:
            logger.error(f"Ошибка при чтении первого файла: {e}")
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении первого файла: {str(e)}")

        # Чтение второго файла (отсканированные QR)
        try:
            df2 = safe_read_excel(file2_data, header=None)
            logger.info(f"Второй файл успешно прочитан. Размер: {df2.shape}")

            if df2.empty:
                raise HTTPException(status_code=400, detail="Второй файл пуст.")

            # Выводим первые несколько строк для отладки
            logger.info(f"Пример данных из второго файла:\n{df2.head().to_string()}")
        except Exception as e:
            logger.error(f"Ошибка при чтении второго файла: {e}")
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении второго файла: {str(e)}")

        # Находим колонки с серийными номерами и QR-кодами в первом файле
        series_column, qr_column1 = find_qr_series_columns(df1)
        logger.info(f"Определены колонки в первом файле: Серия={series_column}, QR={qr_column1}")

        # Находим колонку с QR-кодами во втором файле
        _, qr_column2 = find_qr_series_columns(df2)
        logger.info(f"Определена колонка QR во втором файле: {qr_column2}")

        # Очистка и извлечение серийных номеров из первого файла
        df1['clean_series'] = df1[series_column].apply(clean_code)
        valid_series = df1['clean_series'].dropna()
        series_set = set(valid_series)

        # Очистка и извлечение QR-кодов из первого файла (если есть)
        if qr_column1:
            df1['clean_qr1'] = df1[qr_column1].apply(clean_code)

        # Очистка и извлечение QR-кодов из второго файла
        df2['clean_qr2'] = df2[qr_column2].apply(clean_code)

        # Подсчет общего количества непустых QR-кодов во втором файле
        total_qr_count = df2['clean_qr2'].notna().sum()

        # Получаем уникальные QR-коды
        valid_qr2 = df2['clean_qr2'].dropna()
        qr_set2 = set(valid_qr2)

        logger.info(f"Серийных номеров в первом файле: {len(series_set)} из {len(df1)}")
        logger.info(f"Всего QR-кодов во втором файле: {total_qr_count}")
        logger.info(f"Уникальных QR-кодов во втором файле: {len(qr_set2)}")

        # Логируем несколько примеров для проверки
        if series_set:
            logger.info(f"Примеры серийных номеров из первого файла: {list(series_set)[:5]}")
        if qr_set2:
            logger.info(f"Примеры QR-кодов из второго файла: {list(qr_set2)[:5]}")

        # Создаем словарь для сопоставления серийных номеров и QR-кодов
        series_to_qr = {}
        found_series = set()
        matched_qrs = set()

        # Для каждого QR-кода из второго файла пытаемся найти соответствующий серийный номер
        for qr in qr_set2:
            serial = extract_serial_from_qr(qr, series_set)
            if serial:
                series_to_qr[serial] = qr
                found_series.add(serial)
                matched_qrs.add(qr)

        # Проверяем наличие разницы в серийных номерах и QR-кодах
        series_not_in_qr = series_set - found_series  # Серийные номера, отсутствующие в QR-кодах
        qr_not_matching_series = qr_set2 - matched_qrs  # QR-коды, не соответствующие ни одному серийному номеру

        # Подготавливаем результаты
        series_differences = []
        qr_differences = []
        quantity_diff = ""

        # Формируем строку с информацией о количестве
        quantity_diff = (
            f"ИТОГОВАЯ СТАТИСТИКА:\n"
            f"Серийных номеров в первом файле: {len(series_set)}\n"
            f"Всего QR-кодов во втором файле: {total_qr_count}\n"
            f"Уникальных QR-кодов во втором файле: {len(qr_set2)}\n"
            f"Серийных номеров, найденных в QR-кодах: {len(found_series)}\n"
            f"Серийных номеров, не найденных в QR-кодах: {len(series_not_in_qr)}\n"
            f"QR-кодов, не соответствующих ни одному серийному номеру: {len(qr_not_matching_series)}"
        )
        logger.info(quantity_diff)

        # Добавляем отсутствующие серийные номера в результаты
        for serial in series_not_in_qr:
            series_differences.append(f"Серийный номер: {serial}")

        # Добавляем QR-коды, не соответствующие ни одному серийному номеру
        for qr in qr_not_matching_series:
            qr_differences.append(f"QR-код: {qr}")

        # Формируем файл с результатами если есть различия
        if series_differences or qr_differences or quantity_diff:
            # Генерируем уникальное имя файла с результатами
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]  # Берем первые 8 символов UUID
            filename = f"comparison_{timestamp}_{unique_id}.txt"
            txt_path = os.path.join(STATIC_DIR, filename)

            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(f"{quantity_diff}\n\n")

                if series_differences:
                    txt_file.write("=== СЕРИЙНЫЕ НОМЕРА, ОТСУТСТВУЮЩИЕ В QR-КОДАХ ===\n")
                    txt_file.write("\n".join(series_differences))
                    txt_file.write("\n\n")

                if qr_differences:
                    txt_file.write("=== QR-КОДЫ, НЕ СООТВЕТСТВУЮЩИЕ НИ ОДНОМУ СЕРИЙНОМУ НОМЕРУ ===\n")
                    txt_file.write("\n".join(qr_differences))

            logger.info(f"Создан файл с результатами: {txt_path}")
            return txt_path
        else:
            logger.info("Различий не найдено")
            return None

    except Exception as e:
        logger.error(f"Ошибка при сравнении файлов: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сравнении файлов: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Общее время выполнения: {elapsed_time:.2f} секунд")


def cleanup_old_files(directory=STATIC_DIR, max_files=50, max_age_days=7):
    """
    Очищает старые файлы результатов сравнения.
    Оставляет не более max_files файлов и удаляет файлы старше max_age_days дней.
    """
    try:
        # Получаем список файлов результатов
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.startswith('comparison_') and f.endswith('.txt')]

        if not files:
            return

        # Сортируем файлы по времени изменения (от старых к новым)
        files.sort(key=lambda x: os.path.getmtime(x))

        # Удаляем старые файлы, оставляя максимум max_files
        if len(files) > max_files:
            for file_path in files[:-max_files]:
                try:
                    os.remove(file_path)
                    logger.info(f"Удален старый файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении файла {file_path}: {e}")

        # Удаляем файлы старше max_age_days дней
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        for file_path in files:
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Удален устаревший файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении файла {file_path}: {e}")

    except Exception as e:
        logger.error(f"Ошибка при очистке старых файлов: {e}")



@app.post("/compare/")
async def compare_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Обработка запроса на сравнение файлов"""
    try:
        logger.info(f"Получен запрос на сравнение файлов: {file1.filename} и {file2.filename}")

        # Очистка старых файлов перед созданием новых
        cleanup_old_files()

        # Чтение файлов в память
        file1_data = await file1.read()
        file2_data = await file2.read()

        logger.info(
            f"Файлы успешно загружены. Размер файла 1: {len(file1_data) / 1024:.1f} КБ, файла 2: {len(file2_data) / 1024:.1f} КБ")

        # Сравниваем файлы
        txt_result = await compare_qr_codes(file1_data, file2_data)

        if txt_result is None:
            return {"message": "Различий не найдено! Все серийные номера найдены в QR-кодах."}

        # Извлекаем только имя файла из полного пути
        filename = os.path.basename(txt_result)

        # Проверяем наличие информации о количестве товаров в результате
        has_quantity_diff = False
        try:
            with open(txt_result, "r", encoding="utf-8") as f:
                content = f.read()
                has_quantity_diff = "ИТОГОВАЯ СТАТИСТИКА" in content
        except Exception as e:
            logger.error(f"Ошибка при чтении файла результатов: {e}")

        # Формируем и возвращаем результат
        return {
            "txt_url": f"/static/{filename}",
            "filename": filename,
            "quantity_diff": "Обнаружены расхождения" if has_quantity_diff else ""
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сравнении файлов: {str(e)}")


@app.get("/")
async def get_index():
    """Маршрут для обслуживания index.html"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Сравнение серийных номеров и QR-кодов</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    text-align: center; 
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 30px auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { 
                    color: #2c3e50;
                    margin-bottom: 30px;
                }
                .file-input {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    background-color: #f9f9f9;
                    text-align: left;
                }
                .file-label {
                    display: block;
                    margin-bottom: 10px;
                    font-weight: bold;
                    color: #34495e;
                }
                input[type="file"] {
                    width: 100%;
                    padding: 10px;
                    border: 1px dashed #ccc;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                button {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    margin: 20px 0;
                    cursor: pointer;
                    border-radius: 4px;
                    font-size: 16px;
                    transition: background-color 0.3s;
                }
                button:hover {
                    background-color: #2980b9;
                }
                button:disabled {
                    background-color: #95a5a6;
                    cursor: not-allowed;
                }
                #result {
                    margin-top: 30px;
                    padding: 15px;
                    border-radius: 4px;
                    display: none;
                }
                #result.error {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
                #result.success {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                #loading {
                    display: none;
                    margin: 20px auto;
                }
                .spinner {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #3498db;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 15px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .footer {
                    margin-top: 40px;
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .download-link {
                    display: inline-block;
                    background-color: #27ae60;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 15px;
                    transition: background-color 0.3s;
                }
                .download-link:hover {
                    background-color: #2ecc71;
                }
                #quantity-diff {
                    margin-top: 15px;
                    padding: 10px;
                    background-color: #fff3cd;
                    color: #856404;
                    border: 1px solid #ffeeba;
                    border-radius: 4px;
                    display: none;
                }
                .info-text {
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-top: 5px;
                }
                .instructions {
                    text-align: left;
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 4px;
                    margin: 20px 0;
                    color: #2c3e50;
                }
                .instructions h3 {
                    margin-top: 0;
                    color: #3498db;
                }
                .instructions ul {
                    margin-bottom: 0;
                    padding-left: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Сравнение серийных номеров и QR-кодов</h1>

                <div class="instructions">
                    <h3>Как использовать:</h3>
                    <ul>
                        <li>Загрузите первый файл с серийными номерами товаров</li>
                        <li>Загрузите второй файл с отсканированными QR-кодами</li>
                        <li>Нажмите кнопку "Сравнить файлы"</li>
                        <li>Система проведет полное сравнение и покажет:</li>
                        <ul>
                            <li>Серийные номера, отсутствующие в QR-кодах</li>
                            <li>QR-коды, не соответствующие ни одному серийному номеру</li>
                        </ul>
                    </ul>
                </div>

                <div class="file-input">
                    <div class="file-label">Первый файл (с серийными номерами):</div>
                    <input type="file" id="file1" accept=".xlsx,.xls">
                    <div class="info-text">Excel-файл с серийными номерами товаров</div>
                </div>

                <div class="file-input">
                    <div class="file-label">Второй файл (отсканированные QR-коды):</div>
                    <input type="file" id="file2" accept=".xlsx,.xls">
                    <div class="info-text">Excel-файл с фактически отсканированными QR-кодами</div>
                </div>

                <button id="compare-btn" onclick="uploadFiles()">Сравнить файлы</button>

                <div id="loading">
                    <div class="spinner"></div>
                    <p>Выполняется сравнение файлов...</p>
                    <p class="info-text">Для больших файлов это может занять несколько минут</p>
                </div>

                <div id="result"></div>
                <div id="quantity-diff"></div>

                <div class="footer">
                    Система сравнения серийных номеров и QR-кодов © 2025
                </div>
            </div>

            <script>
                const SERVER_URL = window.location.origin;
                const compareBtn = document.getElementById("compare-btn");
                const file1Input = document.getElementById("file1");
                const file2Input = document.getElementById("file2");

                // Активировать/деактивировать кнопку в зависимости от выбора файлов
                function updateButtonState() {
                    compareBtn.disabled = !(file1Input.files.length > 0 && file2Input.files.length > 0);
                }

                file1Input.addEventListener('change', updateButtonState);
                file2Input.addEventListener('change', updateButtonState);
                updateButtonState(); // Инициализация

                async function uploadFiles() {
                    let file1 = document.getElementById("file1").files[0];
                    let file2 = document.getElementById("file2").files[0];
                    const resultDiv = document.getElementById("result");
                    const quantityDiffDiv = document.getElementById("quantity-diff");
                    const loadingDiv = document.getElementById("loading");
                    const compareBtn = document.getElementById("compare-btn");

                    resultDiv.style.display = "none";
                    quantityDiffDiv.style.display = "none";

                    if (!file1 || !file2) {
                        alert("Пожалуйста, загрузите оба файла!");
                        return;
                    }

                    // Проверка размеров файлов
                    if (file1.size > 50 * 1024 * 1024 || file2.size > 50 * 1024 * 1024) {
                        alert("Размер файла слишком большой (более 50 МБ). Пожалуйста, используйте файлы меньшего размера.");
                        return;
                    }

                    loadingDiv.style.display = "block";
                    compareBtn.disabled = true;

                    let formData = new FormData();
                    formData.append("file1", file1);
                    formData.append("file2", file2);

                    try {
                        let response = await fetch(`${SERVER_URL}/compare/`, {
                            method: "POST",
                            body: formData
                        });

                        if (!response.ok) {
                            let error = await response.json();
                            throw new Error(error.detail || "Произошла ошибка на сервере");
                        }

                        let result = await response.json();
                        resultDiv.style.display = "block";

                        if (result.message) {
                            resultDiv.innerHTML = `<p>${result.message}</p>`;
                            resultDiv.className = "success";
                            quantityDiffDiv.style.display = "none";
                        } else {
                            resultDiv.innerHTML = `
                                ${result.txt_url ? `<a href="${SERVER_URL}${result.txt_url}" class="download-link" download>📥 Скачать файл с результатами сравнения</a>` : ""}
                            `;
                            resultDiv.className = "error";

                            if (result.quantity_diff) {
                                quantityDiffDiv.innerHTML = result.quantity_diff;
                                quantityDiffDiv.style.display = "block";
                            }
                        }
                    } catch (error) {
                        resultDiv.style.display = "block";
                        resultDiv.innerHTML = `<p>Ошибка при обработке файлов: ${error.message}</p>`;
                        resultDiv.className = "error";
                    } finally {
                        loadingDiv.style.display = "none";
                        compareBtn.disabled = false;
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html_content)
    except Exception as e:
        logger.error(f"Ошибка при загрузке index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при загрузке страницы")


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """Маршрут для скачивания файла результатов"""
    file_path = os.path.join(STATIC_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=file_name)
    raise HTTPException(status_code=404, detail="Файл не найден")


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск сервера")
    uvicorn.run(app, host="0.0.0.0", port=8000)