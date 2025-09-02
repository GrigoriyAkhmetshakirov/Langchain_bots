import os

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from preprocess import Preprocessor

class Creator:
    """
    Класс для создания, сохранения и загрузки векторных баз данных FAISS.
    """
    def __init__(self):
        """Инициализация"""
        print('Creator ready')

    def make_splitter(self, cfg):
        """
        Создает и возвращает текстовый сплиттер на основе конфигурации.
        
        Args:
            cfg (dict): Конфигурация сплиттера с ключами:
                - unit (str): "tokens" или другие значения для выбора типа сплиттера
                - chunk_size (int): Размер чанка
                - chunk_overlap (int): Перекрытие между чанками
        
        Returns:
            TextSplitter: Экземпляр текстового сплиттера
        """
        print(cfg)
        if cfg["unit"] == "tokens":
            # Сплиттер на основе токенов
            return TokenTextSplitter(
                encoding_name="cl100k_base",  # Кодировка для подсчета токенов
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"]
            )
        
        # Сплиттер на основе символов
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def get_chunks(self, docs, splitter):
        """
        Разделяет документы на чанки с использованием указанного сплиттера
        
        Args:
            docs (list[Document]): Список документов для обработки
            splitter (TextSplitter): Экземпляр сплиттера для разделения текста
        
        Returns:
            list[Document]: Список чанков-документов
        """
        chunks = []
        for doc in docs:
            # Разделяем текст документа на чанки
            for chunk_text in splitter.split_text(doc.page_content):
                # Копируем метаданные оригинального документа
                md = (doc.metadata or {}).copy() if hasattr(doc, "metadata") else {}
                # Создаем новый документ-чанк
                chunks.append(Document(page_content=chunk_text, metadata=md))

        print(f"Всего чанков={len(chunks)}") 

        return chunks
    
    def create_vector_database(cls, docs, embed_model):
        """
        Создает векторную базу данных FAISS из документов
        
        Args:
            docs (list[Document]): Список документов для индексации
            embed_model: Модель для создания эмбеддингов
        
        Returns:
            FAISS: Векторное хранилище
        """
        db = FAISS.from_documents(docs, embed_model)
        print('База готова')
        return db
    
    def save_database(self, db, path='databases', name='faiss_db'):
        """
        Сохраняет векторную базу на диск
        
        Args:
            db (FAISS): Векторное хранилище для сохранения
            path (str): Путь к директории для сохранения
            name (str): Имя базы данных
        """
        os.makedirs(path, exist_ok=True)  # Создаем директорию, если не существует
        db.save_local(os.path.join(path, name))
        print(f'Векторное хранилище сохранено в: {path}/{name}')

    def load_database(self, embed_model, path='databases', name='faiss_db'):
        """
        Загружает векторную базу с диска
        
        Args:
            embed_model: Модель эмбеддингов (должна быть той же, что при создании)
            path (str): Путь к директории с базой
            name (str): Имя базы данных
        
        Returns:
            FAISS: Загруженное векторное хранилище
        """
        db = FAISS.load_local(
            folder_path=os.path.join(path, name),
            embeddings=embed_model,
            allow_dangerous_deserialization=True
        )
        print(f'Векторное хранилище загружено из: {path}/{name}')
        return db

if __name__ == '__main__':
    # Конфигурации для различных стратегий разделения текста
    configs = [
        {"name": "tok_500_50",   "unit": "tokens", "chunk_size": 500,  "chunk_overlap": 50},
        {"name": "tok_800_100",  "unit": "tokens", "chunk_size": 800,  "chunk_overlap": 100},
        {"name": "tok_1200_0",   "unit": "tokens", "chunk_size": 1200, "chunk_overlap": 0},
        {"name": "chr_1000_100", "unit": "chars",  "chunk_size": 1000, "chunk_overlap": 100},
        {"name": "chr_100_10",   "unit": "chars",  "chunk_size": 100, "chunk_overlap": 10},
    ]
    
    creator = Creator()

    # Предобработка документов
    preprocessed_docs = Preprocessor.preprocess_documents('data')

    # Создание сплиттера по последней конфигурации
    splitter =  creator.make_splitter(configs[-1])

    # Разделение документов на чанки
    chunks = creator.get_chunks(preprocessed_docs, splitter)

    # Инициализация модели для создания эмбеддингов
    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2'
        # "sentence-transformers/all-MiniLM-L6-v2"  # Легкая и быстрая модель
        # "sentence-transformers/all-mpnet-base-v2"  # Более качественная, но медленнее
        # "intfloat/multilingual-e5-small"  # Для мультиязычных данных
    )

    # Создание и сохранение базы
    db = creator.create_vector_database(chunks, embed_model)
    creator.save_database(db)
    
    # Загрузка существующей базы
    db = creator.load_database(embed_model)
    
    # Пример поиска по базе
    question = 'Какие данные нужны для обучения модели'

    # Поиск наиболее релевантных чанков
    docs_and_scores = db.similarity_search_with_score(question, k=3)
    
    # Вывод результатов поиска
    for doc, score in docs_and_scores:
        src = (doc.metadata or {}).get("source", "—")  # Источник документа
        page = (doc.metadata or {}).get("page", "—")   # Номер страницы
        snippet = doc.page_content.replace("\n", " ")   # Текст
        print(f" - Найден фрагмент (score={score:.4f}): {snippet}... [{src} стр.{page}]")