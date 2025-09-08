from dotenv import load_dotenv

import numexpr

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

from langchain.agents import AgentType, initialize_agent, Tool

load_dotenv()

def calculate(expression):
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f'Ошибка: {str(e)}'
    
# Создаем инструмент для агента
calc_tool = Tool(
    name='Calculator',
    func=calculate,
    description='Вычисляет математические выражения. Входные данные должны быть строкой с математическим выражением.'
)

# Инициализация модели
llm = init_chat_model(
    'deepseek-chat',
    model_provider='deepseek',
    temperature=0,
    max_tokens=100,
)

# Создаем агента с инструментами
tools = [calc_tool]

# Системное сообщение для агента
system_message = ChatPromptTemplate('Ты - полезный помощник, который может решать математические задачи. Используй калькулятор для вычислений.')

# Создаем агента
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        'system_message': system_message
    }
)

try:
    result = agent.invoke('75 + 30')
    print('Результат:', result)
except Exception as e:
    print(f'Ошибка: {str(e)}')