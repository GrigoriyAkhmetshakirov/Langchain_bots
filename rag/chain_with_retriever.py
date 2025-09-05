from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from faiss_database_creator import Creator

load_dotenv()

# Модель для эмбеддингов
embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2'
    # 'sentence-transformers/all-MiniLM-L6-v2'  # Легкая и быстрая модель
    # 'sentence-transformers/all-mpnet-base-v2'  # Более качественная, но медленнее
    # 'intfloat/multilingual-e5-small'  # Для мультиязычных данных
)

creator = Creator()

# Загрузка существующей базы
vector_store = creator.load_database(embed_model)

# Определяем ретривер
retriever = vector_store.as_retriever(
    search_type='similarity', 
    search_kwargs={'k': 5}
)

# Запрос с фильтрацией ответов
query = 'Какие данные нужны для обучения модели'
print(f'\nВопрос: {query}')
docs = retriever.invoke(query, filter={'language': 'ru'})
print('\nОтвет ретривера:')
for doc in docs:
    print(doc.metadata, '->', doc.page_content)

#  Цепочка с моделью
llm = init_chat_model(
    'deepseek-chat',
    model_provider='deepseek',
    temperature=0,
    max_tokens=1000,
    # request_timeout=10
)


# Базовый промпт
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",         # метод комбинации документов: "stuff" = просто вставить все
    retriever=retriever,
    return_source_documents=True   # чтобы потом увидеть, какие документы использованы
)

result = qa_chain.invoke({"query": query})
print("\nБазовый ответ:", result["result"])
for i, doc in enumerate(result["source_documents"], 1):
    print(f"Источник {i}: {doc.metadata.get('source')} стр.{doc.metadata.get('page')}")



# Кастомный промпт
template = """Используй следующий контекст для ответа на вопрос. Если ответ не находится в контексте, ответь "Не знаю". Отвечай развернуто и по делу. По возможности цитируй данные из источников. После приведенных фактов указывай в квадратных скобках номер источника.
Контекст: {context}
Ответ:"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

result = chain.invoke({"input": query})
print("\nКастомный ответ:", result['answer'])