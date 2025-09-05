from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset import TestsetGenerator

from preprocess import Preprocessor
from faiss_database_creator import Creator

load_dotenv()

# Создаем чанки для из документо
creator = Creator()
# Предобработка документов
preprocessed_docs = Preprocessor.preprocess_documents('data')

# Создание сплиттера по последней конфигурации
splitter =  creator.make_splitter({'unit': 'chars', 'chunk_size': 400, 'chunk_overlap': 50})

# Разделение документов на чанки
chunks = creator.get_chunks(preprocessed_docs, splitter)

# Модель для получения эмбедингов
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2'
)

# Языковая модель
llm = init_chat_model(
    'deepseek-chat',
    model_provider='deepseek',
    temperature=0,
    max_tokens=1000
)
generator_llm = LangchainLLMWrapper(llm)

# Строим граф знаний
kg = KnowledgeGraph()
for chunk in chunks:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={'page_content': chunk.page_content, 'document_metadata': chunk.metadata}
        )
    )

# print(kg)

# Определяем экстракторы и преобразования
headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)

# Определяем персоны для вопросов
ml = Persona(
    name='Ml-engineer',
    role_description='Ты инженер по машинному обучению. Ты задаешь вопросы по ML части на русском языке',
)

python = Persona(
    name='python developer',
    role_description='Ты питон разработчик. Ты задаешь вопросы по питон части на русском языке',
)

manager = Persona(
    name='Менеджер',
    role_description='Ты Менеджер. Ты задаешь общие вопросы',
)

personas = [ml, python, manager]


# Определяем типы задаваемых вопросов
query_distribution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name='headlines'),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name='keyphrases'
        ),
        0.5,
    ),
]

# Генератор данных
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embedding_model,
    knowledge_graph=kg,
    persona_list=personas,
)

# Генерация и сохранение
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
result = testset.to_pandas()[['user_input', 'reference_contexts', 'reference']]
result.to_csv('results/generated.csv', index=False)