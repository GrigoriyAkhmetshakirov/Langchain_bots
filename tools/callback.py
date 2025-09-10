from dotenv import load_dotenv
import time
import pandas as pd
import os
from datetime import datetime


from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

load_dotenv()


class TokenLoggingCallback(BaseCallbackHandler):
    '''
    Callback для логирования метрик
    Сохраняет в pandas DataFrame + CSV:
    - время запроса
    - запрос
    - ответ
    - входные токены
    - выходные токены
    - кэшированные токены
    - точную стоимость в USD
    - время отклика
    - скорость генерации (токенов/сек)
    '''

    # Тарифы DeepSeek Chat V3 (USD за 1М токенов)
    PRICE_INPUT_CACHE_HIT = 0.07
    PRICE_INPUT_CACHE_MISS = 0.56
    PRICE_OUTPUT = 1.68

    def __init__(self, log_dir: str = 'logs'):
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y-%m-%d')
        self.csv_path = os.path.join(log_dir, f'llm_logs_{date_str}.csv')

        self.logs = pd.DataFrame(columns=[
            'request_time', 'prompt', 'response',
            'input_tokens', 'output_tokens', 'cached_tokens',
            'cost_usd', 'latency_sec', 'tokens_per_sec'
        ])
        self._start_time = None
        self._prompt = None

        # Если CSV уже существует — подгружаем историю
        if os.path.exists(self.csv_path):
            self.logs = pd.read_csv(self.csv_path)

    def _save_csv(self):
        '''Сохраняем DataFrame в CSV'''
        self.logs.to_csv(self.csv_path, index=False, encoding='utf-8')

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start_time = time.time()
        self._prompt = prompts[0] if prompts else None

    def on_llm_end(self, response, **kwargs):
        latency = time.time() - self._start_time
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        usage = response.llm_output.get('token_usage', {}) if response.llm_output else {}
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        cached_tokens = usage.get('cached_tokens', 0)
        cache_hit_tokens = usage['prompt_tokens_details']['cached_tokens']
        cache_miss_tokens = usage['prompt_cache_miss_tokens']

        cost_input = (
            cache_hit_tokens * (self.PRICE_INPUT_CACHE_HIT / 1_000_000) +
            cache_miss_tokens * (self.PRICE_INPUT_CACHE_MISS / 1_000_000)
        )
        cost_output = output_tokens * (self.PRICE_OUTPUT / 1_000_000)
        total_cost = cost_input + cost_output

        text = response.generations[0][0].text if response.generations else None
        total_tokens = input_tokens + output_tokens
        tps = total_tokens / latency if latency > 0 else None

        row = {
            'request_time': request_time,
            'prompt': self._prompt,
            'response': text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'cost_usd': total_cost,
            'latency_sec': latency,
            'tokens_per_sec': tps
        }
        self.logs = pd.concat([self.logs, pd.DataFrame([row])], ignore_index=True)

        # Сохраняем после каждого запроса
        self._save_csv()


if __name__ == '__main__':
    # Создаём колбэк
    cb = TokenLoggingCallback(log_dir='logs')

    # Инициализируем LLM
    llm = init_chat_model(
        'deepseek-chat',
        model_provider='deepseek',
        temperature=0,
        max_tokens=2000,
        callbacks=[cb],
        verbose=False,
    )
    template = 'Ты - интеллектуальный ассистент, который помогает пользователям, отвечая на их вопрос. Отвечай кратко одним предложением. Вопрос: {question}'

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm 

    questions = [
        'Привет, Что такое машинное обучение?',
        'Объясни простыми словами, как работает трансформер.',
        'Сколько будет 123456 * 789?'
    ]

    for q in questions:
        resp = chain.invoke({'question': q})
