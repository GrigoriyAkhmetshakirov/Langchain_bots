from dotenv import load_dotenv

from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model

load_dotenv()

# Инициализация модели
llm = init_chat_model(
    'deepseek-chat',
    model_provider='deepseek',
    temperature=0,
    max_tokens=2000,
    verbose=True,
)

# Экспертные цепочки
experts = {
    'math': ChatPromptTemplate.from_template(
        'Ты - калькулятор. Дай только число, всегда представляйся: {question}'
    ) | llm,
    'explain': ChatPromptTemplate.from_template(
        'Ты - преподаватель. Объясни одним предложением, всегда представляйся: {question}'
    ) | llm,
    'default': ChatPromptTemplate.from_template(
        'Ты - зоолог. Отвечай одним предложением, всегда представляйся: {question}'
    ) | llm,
}

# Роутер
route_system = (
    'Определи, к какому эксперту направить вопрос пользователя: '
    '"math", "explain" или "default. '
    'Ответь строго одним словом: math, explain или default.'
)
route_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', route_system),
        ('human', '{question}'),
    ]
)

class RouteQuery(TypedDict):
    '''Route query to destination.'''
    destination: Literal['math', 'explain', 'default']

# Цепочка роутинга
route_chain = (
    route_prompt
    | llm.with_structured_output(RouteQuery)
    | itemgetter('destination')
)

# Финальная цепочка
router_chain = (
    {
        'destination': route_chain,  # выбор эксперта
        'question': lambda x: x['question'],  # passthrough
    }
    | RunnableLambda(
        lambda x: experts[x['destination']].invoke({'question': x['question']})
    )
)

questions = [
    'Привет, что такое машинное обучение?',
    'Что такое кошка?',
    'Сколько будет 123456 * 789?'
]

for q in questions:
    answer = router_chain.invoke({'question': q})
    print(f'Q: {q}\nA: {answer.content}\n')
