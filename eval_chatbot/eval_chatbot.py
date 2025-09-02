from dotenv import load_dotenv
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import statistics

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

class EvalBot:

    def __init__(self):
        base_path = Path('templates')

        # Загружаем конфигурацию
        self.config = self._load_yaml(base_path/'config_eval.yaml')
        self.system_prompt_config = self._load_yaml(base_path/'system_prompt_eval.yaml')
        
        # Указываем путь к JSON-файлу с результатами
        self.eval_results_path = Path('results/eval_results.json')
        
        # Создаем директорию если не существует
        self.eval_results_path.parent.mkdir(exist_ok=True)

        # Формируем системный промпт
        self.system_template, self.output_parser = self._build_system_prompt()

        # Инициализация модели
        self.llm = init_chat_model(
            'deepseek-chat',
            model_provider='deepseek',
            temperature=0.1,
            max_tokens=1000,
            # request_timeout=10
        )

        # Создание цепочки
        self.chain = self._create_chain()

    def _load_yaml(self, file_path):
        '''Загрузка YAML-файла'''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Возвращаем пустой словарь, если файл не найден
            return {}

    def _build_system_prompt(self):
        '''Создание и форматирование системного промпта'''

        # Определяем текущую версию шаблона
        version = self.system_prompt_config['prompts']['support_answer']['current']
        template = self.system_prompt_config['prompts']['support_answer']['versions'][version]

        formatted_prompt = template.format()

        # Формируем схему ответа
        schemas = [
            ResponseSchema(name=field, description=desc, type='dict')
            for field, desc in self.config['eval_format']['fields'].items()
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
                chat_history=lambda x: x.get('chat_history', [])
            )
            | prompt
            | self.llm
            | self.output_parser
        )

    def eval_messages(self, messages):
        report_data = {'answers': [], 'summary': []}
        summary = defaultdict(int)
        for i, message in tqdm(enumerate(messages), total=len(messages), desc='Анализ'):
            try:
                response = self.chain.invoke({'text': message})
                answer_entry = {
                    'id': i+1,
                    'message': message,
                    'response': response
                }
                report_data['answers'].append(answer_entry)

                for key, value in response['eval_results'].items():
                    summary[key] += value

            except Exception as e:
                print(f'Ошибка: {e}')
                return 'Возникла ошибка при обработке запроса'
            
        
        # Итоговые оценки
        summary = {k: v/len(messages) for k,v in summary.items()}
        summary['harmonic_mean'] = statistics.harmonic_mean(list(summary.values()), weights=None)
        report_data['summary'].append(summary)

        # Записываем результат в JSON-файл
        with open(self.eval_results_path, 'w', encoding='utf-8') as json_file:
            json.dump(report_data, json_file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    bot = EvalBot()
    messages = ['Добрый день!', 'чем могу помочь?', 'че каво'] 
    result = bot.eval_messages(messages)