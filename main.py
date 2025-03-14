from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from io import BytesIO
import os
import logging
import re


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


def clean_code(value):
    """Очистка QR-кода или серии от лишних символов"""
    if pd.isna(value) or value is None:
        return None
    # Удаляем пробелы, кавычки и переводы строк
    return str(value).strip().strip('"').strip("'").replace('\n', '').replace('\r', '')


def is_valid_qr(value):
    """Проверка валидности QR-кода (должен содержать много цифр)"""
    if not value or pd.isna(value):
        return False
    value_str = str(value).strip()
    # Строка должна быть в основном из цифр и достаточно длинной для QR-кода
    digit_count = sum(c.isdigit() for c in value_str)
    return digit_count >= 10 and len(value_str) >= 15


def is_valid_series(value):
    """Проверка валидности серии (6-8 цифр)"""
    if not value or pd.isna(value):
        return False
    value_str = str(value).strip()
    # Серия обычно состоит из 6-8 цифр
    return len(value_str) >= 4 and len(value_str) <= 12 and sum(c.isdigit() for c in value_str) >= 4


def find_qr_series_columns(df):
    """Находит колонки с QR-кодами и сериями в DataFrame"""
    qr_column = None
    series_column = None

    # 1. Сначала ищем по названиям колонок
    for col in df.columns:
        col_str = str(col).lower()

        # Ищем колонку с QR-кодами по имени
        if any(keyword in col_str for keyword in ['qr', 'код', 'штрих', 'штрихкод', 'datamatrix']):
            qr_column = col
            logger.info(f"Найдена колонка QR по имени: {col}")

        # Ищем колонку с сериями по имени
        if any(keyword in col_str for keyword in ['серия', 'серии', 'series', 'партия']):
            series_column = col
            logger.info(f"Найдена колонка Серия по имени: {col}")

    # 2. Если по имени не нашли, ищем по содержимому
    if qr_column is None:
        qr_counts = {}
        for col in df.columns:
            valid_qrs = df[col].apply(is_valid_qr).sum()
            if valid_qrs > 0:
                qr_counts[col] = valid_qrs

        if qr_counts:
            # Выбираем колонку с наибольшим количеством QR-кодов
            qr_column = max(qr_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Найдена колонка QR по содержимому: {qr_column} ({qr_counts[qr_column]} QR-кодов)")

    # 3. Если колонку с сериями не нашли по имени, ищем по содержимому
    if series_column is None:
        series_counts = {}
        for col in df.columns:
            if col != qr_column:  # Исключаем колонку с QR
                valid_series = df[col].apply(is_valid_series).sum()
                if valid_series > 0:
                    series_counts[col] = valid_series

        if series_counts:
            # Выбираем колонку с наибольшим количеством серий
            series_column = max(series_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Найдена колонка Серия по содержимому: {series_column} ({series_counts[series_column]} серий)")

    # 4. Если все еще не нашли QR, используем первую колонку
    if qr_column is None and len(df.columns) > 0:
        qr_column = df.columns[0]
        logger.info(f"QR-колонка не найдена, используем первую колонку: {qr_column}")

    return qr_column, series_column


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


async def compare_qr_codes(file1_data, file2_data):
    """Улучшенное сравнение QR-кодов из двух файлов"""
    try:
        logger.info("Начинаем сравнение файлов")

        # Чтение первого файла (из ДЛО)
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

        # Находим колонки с QR-кодами и сериями
        qr_column1, series_column = find_qr_series_columns(df1)
        logger.info(f"Определены колонки в первом файле: QR={qr_column1}, Серия={series_column}")

        # Для второго файла находим колонку с QR-кодами
        qr_column2, _ = find_qr_series_columns(df2)
        logger.info(f"Определена колонка QR во втором файле: {qr_column2}")

        # Очистка и подготовка QR-кодов
        df1['clean_qr'] = df1[qr_column1].apply(clean_code)
        df2['clean_qr'] = df2[qr_column2].apply(clean_code)

        # Фильтрация валидных QR-кодов
        valid_qr1 = df1[df1['clean_qr'].apply(is_valid_qr)]
        valid_qr2 = df2[df2['clean_qr'].apply(is_valid_qr)]

        logger.info(f"Валидных QR в первом файле: {len(valid_qr1)} из {len(df1)}")
        logger.info(f"Валидных QR во втором файле: {len(valid_qr2)} из {len(df2)}")

        # Получаем уникальные QR-коды
        qr_set1 = set(valid_qr1['clean_qr'].dropna())
        qr_set2 = set(valid_qr2['clean_qr'].dropna())

        logger.info(f"Уникальных QR в первом файле: {len(qr_set1)}")
        logger.info(f"Уникальных QR во втором файле: {len(qr_set2)}")

        # Логируем несколько примеров QR-кодов для проверки
        if qr_set1:
            logger.info(f"Примеры QR из первого файла: {list(qr_set1)[:5]}")
        if qr_set2:
            logger.info(f"Примеры QR из второго файла: {list(qr_set2)[:5]}")

        # Сравниваем QR-коды
        missing_in_df2 = qr_set1 - qr_set2  # Отсутствуют во втором файле
        missing_in_df1 = qr_set2 - qr_set1  # Отсутствуют в первом файле

        # Подготавливаем результаты
        differences_qr = []
        series_differences = []
        quantity_diff = ""

        if len(qr_set1) != len(qr_set2):
            quantity_diff = f"Количество товаров в файлах различается: В файле 1: {len(qr_set1)} | В файле 2: {len(qr_set2)}"
            logger.info(quantity_diff)

        # Добавляем отсутствующие QR-коды в результаты
        for qr in missing_in_df2:
            differences_qr.append(f"Отправить Поставщику QR: {qr}")

        for qr in missing_in_df1:
            differences_qr.append(f"Поставщик должен нам отправить в ЭБД QR: {qr}")

        # Проверяем серии в QR-кодах, если есть различия и есть колонка с сериями
        if series_column and (missing_in_df1 or missing_in_df2):
            logger.info("Проверяем серии внутри QR-кодов...")

            # Получаем серии из первого файла (исключаем пустые)
            series_data = valid_qr1[series_column].apply(clean_code).dropna()
            valid_series = set(s for s in series_data if s and is_valid_series(s))

            logger.info(f"Найдено {len(valid_series)} уникальных серий в первом файле")
            if valid_series:
                logger.info(f"Примеры серий: {list(valid_series)[:5]}")

            # Проверяем каждую серию в QR-кодах второго файла
            for series in valid_series:
                found = False
                for qr in qr_set2:
                    if series in str(qr):
                        found = True
                        break

                if not found:
                    series_differences.append(f"Серия не найдена в QR: {series}")

            # Проверяем QR-коды из второго файла на наличие серий из первого
            for qr in qr_set2:
                found = False
                for series in valid_series:
                    if series in str(qr):
                        found = True
                        break

                if not found and qr not in missing_in_df1:  # Исключаем уже отмеченные QR
                    series_differences.append(f"QR не содержит серий из первого файла: {qr}")

        # Формируем файл с результатами если есть различия
        if differences_qr or series_differences or quantity_diff:
            txt_path = os.path.join(STATIC_DIR, "filtered_differences.txt")
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                if quantity_diff:
                    txt_file.write(f"{quantity_diff}\n\n")

                if differences_qr:
                    txt_file.write("QR-коды, которые необходимо обработать:\n")
                    txt_file.write("\n".join(differences_qr))
                    txt_file.write("\n\n")

                if series_differences:
                    txt_file.write("Серии, которых нет в QR:\n")
                    txt_file.write("\n".join(series_differences))

            logger.info(f"Создан файл с результатами: {txt_path}")
            return txt_path
        else:
            logger.info("Различий не найдено")
            return None

    except Exception as e:
        logger.error(f"Ошибка при сравнении файлов: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сравнении файлов: {str(e)}")


@app.post("/compare/")
async def compare_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Обработка запроса на сравнение файлов"""
    try:
        logger.info(f"Получен запрос на сравнение файлов: {file1.filename} и {file2.filename}")

        # Чтение файлов в память
        file1_data = await file1.read()
        file2_data = await file2.read()

        logger.info(
            f"Файлы успешно загружены. Размер файла 1: {len(file1_data) / 1024:.1f} КБ, файла 2: {len(file2_data) / 1024:.1f} КБ")

        # Сравниваем файлы
        txt_result = await compare_qr_codes(file1_data, file2_data)

        if txt_result is None:
            return {"message": "Различий не найдено"}

        # Проверяем наличие информации о количестве товаров в результате
        has_quantity_diff = False
        try:
            with open(txt_result, "r", encoding="utf-8") as f:
                content = f.read()
                has_quantity_diff = "Количество товаров в файлах различается" in content
        except Exception as e:
            logger.error(f"Ошибка при чтении файла результатов: {e}")

        # Формируем и возвращаем результат
        return {
            "txt_url": f"/static/{os.path.basename(txt_result)}",
            "quantity_diff": "Количество товаров в файлах различается" if has_quantity_diff else ""
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сравнении файлов: {str(e)}")


@app.get("/")
async def get_index():
    """Маршрут для обслуживания index.html"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        else:
            # Если файла нет, создаем базовую HTML-страницу
            html_content = """
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Приход ЭБД</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        text-align: center; 
                        margin: 50px; 
                    }
                    .container {
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                    h2 { margin-bottom: 20px; }
                    input, button { 
                        margin: 10px; 
                        padding: 8px;
                    }
                    button {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        cursor: pointer;
                        border-radius: 4px;
                    }
                    #result { 
                        margin-top: 30px;
                        padding: 10px;
                        border-radius: 4px;
                    }
                    #result.error {
                        background-color: #f8d7da;
                        color: #721c24;
                    }
                    #result.success {
                        background-color: #d4edda;
                        color: #155724;
                    }
                    .file-input {
                        border: 1px solid #ddd;
                        padding: 10px;
                        margin: 10px 0;
                        border-radius: 4px;
                    }
                    .file-label {
                        display: block;
                        margin-bottom: 5px;
                        text-align: left;
                        font-weight: bold;
                    }
                    #loading {
                        display: none;
                        margin-top: 20px;
                    }
                    .spinner {
                        border: 4px solid #f3f3f3;
                        border-top: 4px solid #3498db;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        animation: spin 1s linear infinite;
                        margin: 0 auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Загрузка файлов для сравнения QR-кодов</h2>

                    <div class="file-input">
                        <div class="file-label">Первый файл (из ДЛО):</div>
                        <input type="file" id="file1" accept=".xlsx,.xls">
                    </div>

                    <div class="file-input">
                        <div class="file-label">Второй файл (отсканированные QR):</div>
                        <input type="file" id="file2" accept=".xlsx,.xls">
                    </div>

                    <button onclick="uploadFiles()">Сравнить файлы</button>

                    <div id="loading">
                        <div class="spinner"></div>
                        <p>Выполняется сравнение файлов...</p>
                    </div>

                    <div id="result" style="display:none;"></div>
                    <div id="quantity-diff" class="quantity-difference" style="display:none;"></div>
                </div>

                <script>
                    const SERVER_URL = window.location.origin;

                    async function uploadFiles() {
                        let file1 = document.getElementById("file1").files[0];
                        let file2 = document.getElementById("file2").files[0];
                        const resultDiv = document.getElementById("result");
                        const quantityDiffDiv = document.getElementById("quantity-diff");
                        const loadingDiv = document.getElementById("loading");

                        resultDiv.style.display = "none";
                        quantityDiffDiv.style.display = "none";

                        if (!file1 || !file2) {
                            alert("Пожалуйста, загрузите оба файла!");
                            return;
                        }

                        loadingDiv.style.display = "block";

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
                                    ${result.txt_url ? `<a href="${SERVER_URL}${result.txt_url}" download>📥 Скачать различия (TXT)</a>` : ""}
                                `;
                                resultDiv.className = "error";

                                if (result.quantity_diff) {
                                    quantityDiffDiv.innerHTML = result.quantity_diff;
                                    quantityDiffDiv.style.display = "block";
                                }
                            }
                        } catch (error) {
                            resultDiv.style.display = "block";
                            resultDiv.innerHTML = `<p>Ошибка при загрузке файлов: ${error.message}</p>`;
                            resultDiv.className = "error";
                        } finally {
                            loadingDiv.style.display = "none";
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