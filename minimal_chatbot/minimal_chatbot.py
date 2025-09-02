from dotenv import load_dotenv
import yaml
import json
from pathlib import Path
import os

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import logging

load_dotenv()

# Отключаем логирование HTTP
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

os.makedirs('logs', exist_ok=True)
# Настройка логгера
logging.basicConfig(
    filename='logs/chat_session.log', 
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info('=== New session ===')

class SupportBot:

    def __init__(self):
        '''Инициализация'''
        base_path = Path('templates')

        # Загружаем конфигурацию
        self.config = self._load_yaml(base_path/'config.yaml')
        self.system_prompt_config = self._load_yaml(base_path/'system_prompt.yaml')
        self.few_shot_examples = self._load_jsonl(base_path/'few_shots.jsonl')

        # Формируем системный промпт
        self.system_template, self.output_parser = self._build_system_prompt()

        # Инициализация модели
        self.llm = init_chat_model(
            'deepseek-chat',
            model_provider='deepseek',
            temperature=0.1,
            max_tokens=500,
            request_timeout=10
        )

        # Создание памяти
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            input_key='text',
            output_key='output',
            max_token_limit=1000,
        )

        # Создание цепочки
        self.chain = self._create_chain()

    def _load_yaml(self, file_path):
        '''Загрузка YAML-файла'''
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_jsonl(self, file_path):
        '''Загрузка JSONL с few-shot примерами'''
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
            print(f'Загружено {len(examples)} few-shot примеров')
        except FileNotFoundError:
            print('Файл few_shots.jsonl не найден, продолжаем без примеров')
        return examples

    def _build_system_prompt(self):
        '''Создание и форматирование системного промпта'''
        
        # Определяем текущую версию шаблона
        version = self.system_prompt_config['prompts']['support_answer']['current']
        template = self.system_prompt_config['prompts']['support_answer']['versions'][version]

        # Подставляем значения из config.yaml
        brand = self.config['brand']
        tone = self.config['tone']
        fallback = self.config['fallback']

        formatted_prompt = template.format(
            brand_name=brand['name'],
            tone_persona=tone['persona'],
            tone_sentences_max=tone['sentences_max'],
            tone_bullets='Используй маркированные списки где уместно' if tone['bullets'] else 'Не используй маркированные списки',
            tone_avoid=', '.join(tone['avoid']),
            tone_must_include=', '.join(tone['must_include']),
            fallback_no_data=fallback['no_data']
        )

        # Добавляем few-shot примеры
        if self.few_shot_examples:
            formatted_prompt += '\n\nПримеры правильных ответов:'
            for i, ex in enumerate(self.few_shot_examples, 1):
                formatted_prompt += f'\n\nПример {i}:\nВопрос: {ex['user']}\nОтвет: {ex['assistant']}'

        # Формируем схему ответа
        schemas = [
            ResponseSchema(name=field, description=desc)
            for field, desc in self.config['format']['fields'].items()
        ]
        output_parser = StructuredOutputParser.from_response_schemas(schemas)

        # Экранируем фигурные скобки
        safe_format = output_parser.get_format_instructions().replace('{', '{{').replace('}', '}}')
        formatted_prompt += f'\n\nФормат ответа:\n{safe_format}'

        print(f'System prompt:\n{formatted_prompt}')
        return formatted_prompt, output_parser

    def _create_chain(self):
        '''Создание цепочки LangChain'''
        prompt = ChatPromptTemplate.from_messages([
            ('system', self.system_template),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{text}')
        ])

        return (
            RunnablePassthrough.assign(
                chat_history=lambda _: self.memory.load_memory_variables({})['chat_history']
            )
            | prompt
            | self.llm
            | self.output_parser
        )

    def process_message(self, message):
        '''Обработка сообщения пользователя'''
        try:
            logging.info(f'User: {message.strip()}')
            response = self.chain.invoke({'text': message})
            for key in response.keys():
                logging.info(f'Bot: {key} - {response[key]}') 
            res = response['answer'] + ' ' + response['actions']
            # Сохраняем историю
            self.memory.save_context(
                {'text': message},
                {'output': json.dumps(res, ensure_ascii=False, indent=2)}
            )
            res = response['answer'] + ' ' + response['actions']
            logging.info(f'Bot full answer: {res}') 
            logging.info('') 
            return res
        except Exception as e:
            print(e)
            logging.error(f'Warning: {e}')
            logging.info('')
            return 'Возникла ошибка при обработке запроса'

    def run_interactive(self):
        '''Запуск в интерактивном режиме'''
        print('Чат-бот запущен. Введите "выход" для завершения.')
        while True:
            user_input = input('Пользователь: ')
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            answer = self.process_message(user_input)
            print(f'Бот: {answer}')

if __name__ == '__main__':
    bot = SupportBot()
    bot.run_interactive()
