import os
import shutil
from dotenv import load_dotenv
from git import Repo
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

class RepoAnalyzer:
    def __init__(self, repo_url, clone_dir='./temp_repo'):
        self.repo_url = repo_url
        self.clone_dir = clone_dir
        self.output_file = os.path.join('results', self.repo_url.split('/')[-1])
        self.llm = init_chat_model(
            'deepseek-chat',
            model_provider='deepseek',
            temperature=0,
            max_tokens=2000,
        )

    def clone_repository(self):
        if os.path.exists(self.clone_dir):
            shutil.rmtree(self.clone_dir)
        print('Клонирование репозитория...')
        Repo.clone_from(self.repo_url, self.clone_dir)

    def extract_code_files(self):
        allowed_extensions = ['.py', '.txt', '.md']
        max_file_size = 1000000  # 1000000 килобайт максимальный размер файла
        
        docs = []
        for root, _, files in os.walk(self.clone_dir):
            for file in files:
                if any(file.endswith(ext) for ext in allowed_extensions):
                    path = os.path.join(root, file)
                    try:
                        # Проверяем размер файла
                        if os.path.getsize(path) > max_file_size:
                            print(f'Пропускаем большой файл: {path}')
                            continue
                            
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if content.strip():  # Пропускаем пустые файлы
                            rel_path = os.path.relpath(path, self.clone_dir)
                            docs.append(Document(
                                page_content=content, 
                                metadata={'source': rel_path}
                            ))
                    except UnicodeDecodeError:
                        print(f'Ошибка чтения файла: {path}')
                    except Exception as e:
                        print(f'Ошибка обработки файла {path}: {str(e)}')
        
        print(f'Собрано документов: {len(docs)}')
        return docs

    def generate_summary(self, docs):
        if not docs:
            return 'Не удалось извлечь файлы для анализа'
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
        print('Разделение документов на чанки...')
        split_docs = text_splitter.split_documents(docs)
        print(f'Получено чанков: {len(split_docs)}')

        map_template = '''Проанализируй следующий код из репозитория:

            {text}

            Создай краткое описание этого фрагмента на русском языке.
        '''

        combine_template = '''На основе следующих описаний частей проекта:

            {text}

            Создай итоговую сводку проекта на русском языке, включая:
            1. Основную цель проекта
            2. Используемые технологии и языки программирования
            3. Ключевые особенности архитектуры и реализации
            4. Общую оценку сложности и масштаба проекта

            Сводка должна быть краткой и информативной.
        '''

        map_prompt = PromptTemplate(template=map_template, input_variables=['text'])
        combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'])

        summary_chain = load_summarize_chain(
            llm=self.llm,
            chain_type='map_reduce',
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )

        print('Генерация сводки...')
        result = summary_chain.invoke(split_docs)
        return result['output_text']

    def save_summary(self, summary):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f'# Анализ репозитория {self.repo_url}\n\n')
            f.write(summary)
        print(f'Результат сохранен в: {self.output_file}')

    def analyze(self):
        try:
            self.clone_repository()
            docs = self.extract_code_files()
            
            if not docs:
                return 'Не найдено файлов для анализа'
                
            summary = self.generate_summary(docs)
            self.save_summary(summary)
            return summary
            
        except Exception as e:
            print(f'Ошибка при анализе: {str(e)}')
            return f'Ошибка: {str(e)}'
        finally:
            if os.path.exists(self.clone_dir):
                shutil.rmtree(self.clone_dir)

if __name__ == '__main__':
    analyzer = RepoAnalyzer('https://github.com/GrigoriyAkhmetshakirov/SPF')
    result = analyzer.analyze()