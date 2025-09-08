from dotenv import load_dotenv
import pandas as pd   
from tqdm import tqdm  

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
  
from ragas.metrics import (  
    context_recall,  
    context_precision,  
    answer_relevancy,  
    faithfulness,
    context_entity_recall,
      
)  
from ragas import evaluate
from ragas import EvaluationDataset

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

#  Цепочка с моделью
llm = init_chat_model(
    'deepseek-chat',
    model_provider='deepseek',
    temperature=0,
    max_tokens=1500,
    # request_timeout=10
)

# Цепочка
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

testset_pd = pd.read_csv('results/generated.csv')

# Запускаем цепочку и сохраняем ответ и контексты
answers = []
for question in tqdm(testset_pd['user_input']):
    answers.append(qa_chain.invoke(question))
testset_pd['response'] = [x['result'] for x in answers]
testset_pd['retrieved_contexts'] = [[doc.page_content for doc in answer['source_documents']] for answer in answers]

testset_pd['reference_contexts'] = testset_pd['reference_contexts'].apply(lambda x: [x.strip("['']")])


# Создание EvaluationDataset
eval_ds = EvaluationDataset.from_pandas(testset_pd)

results = evaluate(eval_ds, 
    metrics=[
        context_recall, 
        context_precision,
        context_entity_recall,
        answer_relevancy, 
        faithfulness
    ],
    llm=llm,
    embeddings=embed_model

)
print('\n', results)

print('\n', 'Анализ результатов:')
# Создаем промпт для анализа метрик
analysis_template = """
Ты - эксперт по оценке RAG-систем. Проанализируй следующие метрики и предоставь развернутый анализ на русском языке:

{results}

Проанализируй:
1. Общую производительность системы
2. Сильные и слабые стороны
3. Рекомендации по улучшению
4. Интерпретацию каждого показателя в контексте RAG-систем

Ответ предоставь в формате профессионального отчета.
"""

prompt = ChatPromptTemplate.from_template(analysis_template)

# Создаем цепочку
analysis_chain = (
    prompt
    | llm 
    | StrOutputParser()
)

# Функция для форматирования метрик в читаемый вид
def format_metrics(results):
    return "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])

# Запускаем цепочку
result = analysis_chain.invoke(results)

print(result)