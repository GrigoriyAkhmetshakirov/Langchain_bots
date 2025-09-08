from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentType, initialize_agent, Tool

load_dotenv()

def create_search_agent():
    # Инициализация LLM
    llm = init_chat_model(
        'deepseek-chat',
        model_provider='deepseek',
        temperature=0,
        max_tokens=1000,
    )
    
    # Настройка поиска
    search = DuckDuckGoSearchResults(region="ru-ru", time="d", max_results=3)
    
    tools = [
        Tool(
            name='web_search',
            func=search.run,
            description='Поиск информации в интернете'
        )
    ]

    # Создаем системный промпт
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — поисковый бот. Всегда используй tool для поиска информации, даже если сам знаешь ответ."),
        ("human", "{input}")
    ])

    # Инициализация агента
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=1,
        early_stopping_method="generate",
        system_message=system_prompt
    )
    
    return agent

if __name__ == '__main__':
    agent = create_search_agent()
    query = 'Акции Apple сегодня'
    res = agent.invoke({"input": query, "chat_history": []})
    print(res)