import glob
import os
import json
import re
import hashlib
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

class Preprocessor:
    """
    Класс для предварительной обработки PDF-документов.
    Выполняет загрузку, очистку текста и фильтрацию документов.
    """
    
    def __init__(self):
        """Базовый конструктор"""
        print('Preprocesser ready')

    @staticmethod
    def clean_text(text):
        """
        Очищает текст от лишних символов и форматирования
        
        Параметры:
            text (str): Исходный текст для очистки
            
        Возвращает:
            str: Очищенный текст
        """
        # Удаление шаблона
        text = re.sub(r'\.{3,}Эта страница в последний раз была отредактирована \.{3,}', ' ', text)
        # Удаление лишних переносов строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Удаление пробелов перед переносами
        text = re.sub(r'[ \t]+\n', '\n', text)
        # Замена одиночных переносов на пробелы
        text = re.sub(r'\n', ' ', text)
        # Удаление множественных пробелов
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    @staticmethod
    def filter_docs(docs):
        """
        Фильтрует список документов, удаляя дубликаты и короткие тексты
        
        Параметры:
            docs (list): Список объектов Document для фильтрации
            
        Возвращает:
            list: Отфильтрованный список документов
        """
        unique = {}  # Хеш-таблица для отслеживания уникальных документов
        filtered = []  # Результирующий список документов
        
        for doc in docs:
            text = doc.page_content
            # Создание хеша для проверки уникальности
            h = hashlib.md5(text.encode('utf-8')).hexdigest()
            if h in unique:
                continue  # Пропуск дубликатов
            unique[h] = True
            
            if len(text) < 30:
                continue  # Пропуск коротких текстов
            filtered.append(doc)
        return filtered
    
    @classmethod
    def preprocess_document(cls, file, len_filter=30):
        """
        Обрабатывает отдельный PDF-файл: извлекает текст и метаданные
        
        Параметры:
            file (str): Путь к PDF-файлу
            len_filter (int): Минимальная длина текста страницы (символов)
            
        Возвращает:
            list: Список объектов Document с обработанными страницами
        """
        # Загрузка и разделение PDF на страницы
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        total_pages = len(pages)
        processed_pages = []
        
        # Обработка каждой страницы документа
        for page_number, page in enumerate(pages, start=1):
            raw_text = page.page_content
            clean_page = cls.clean_text(raw_text)  # Очистка текста
            # Формирование метаданных
            metadata = {
                'source': file.split('/')[-1].removesuffix('.pdf'),  # Имя файла без расширения
                'page': page_number,  # Номер текущей страницы
                'total_pages': total_pages,  # Общее количество страниц
                'language': 'ru'  # Язык документа
            }
            
            # Добавление страницы, если текст соответствует критерию длины
            if len(clean_page) >= len_filter:
                doc = Document(page_content=clean_page, metadata=metadata)
                processed_pages.append(doc)
        
        return processed_pages

    @classmethod
    def preprocess_documents(cls, input_folder):
        """
        Обрабатывает все PDF-файлы в указанной папке.
        
        Параметры:
            input_folder (str): Путь к папке с PDF-файлами
            
        Возвращает:
            list: Отфильтрованный список объектов Document
        """
        docs = []
        # Поиск всех PDF-файлов в целевой папке
        files = glob.glob(f'{input_folder}/*.pdf')
        
        # Обработка файлов
        for file in tqdm(files, desc='Подготовка документов', total=len(files)):
            doc_list = cls.preprocess_document(file)
            docs.extend(doc_list)
            
        filtered = cls.filter_docs(docs)  # Фильтрация документов
        return filtered
    
    @classmethod
    def dump_to_json(cls, preprocessed_docs, path='results'):
        """
        Сохраняет предобработанные документы в JSON-фай
        
        Параметры:
            preprocessed_docs (list): Список предобработанных объектов Document
            path (str): Путь к директории для сохранения результатов (по умолчанию 'results')
            
        Возвращает:
            None
        """
        # Создание директории для результатов (если не существует)
        os.makedirs(path, exist_ok=True)
        
        # Подготовка данных для экспорта
        export = [{'text': d.page_content, **d.metadata} for d in preprocessed_docs]
        
        # Сохранение в JSON-файл
        with open(os.path.join(path, 'preprocessed_data.json'), 'w', encoding='utf-8') as f:
            json.dump(export, f, ensure_ascii=False, indent=2)


if __name__  == '__main__':
    # Пример использования: обработка документов в папке 'data'
    preprocessed_docs = Preprocessor.preprocess_documents('data')
    print(f'Total documents after preprocessing: {len(preprocessed_docs)}')
    
    # Экспорт результатов в JSON-файл
    Preprocessor.dump_to_json(preprocessed_docs)