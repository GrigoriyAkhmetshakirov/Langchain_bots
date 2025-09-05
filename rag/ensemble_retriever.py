from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from preprocess import Preprocessor
from faiss_database_creator import Creator

class HybridRetriever:
    """Класс для гибридного поиска с объединением векторного и BM25 поиска"""
    
    def __init__(self, chunks, top_n=5):
        """
        Инициализация гибридного ретривера
        
        Args:
            chunks: список чанков для поиска
        """
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        self._initialize_retrievers()
        self.top_n = top_n
    
    def _initialize_retrievers(self, k=15):
        """
        Инициализация векторного и BM25 ретриверов
        
        Args:
            k: количество возвращаемых кандидатов
        """
        # Векторный ретривер (FAISS)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={'k': k}
        )
        
        # Текстовый ретривер (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = k
    
    @staticmethod
    def reciprocal_rank_fusion(results, k=60):
        """
        Reciprocal Rank Fusion algorithm для объединения результатов
        
        Args:
            results: список результатов от разных ретриверов
            k: константа для сглаживания
            
        Returns:
            Отсортированный список документов
        """
        scores = {}
        doc_map = {}
        
        for retriever_results in results:
            for rank, doc in enumerate(retriever_results):
                doc_content = doc.page_content
                doc_map[doc_content] = doc
                
                if doc_content not in scores:
                    scores[doc_content] = 0
                scores[doc_content] += 1 / (rank + k)
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]
    
    @staticmethod
    def merge_retriever_results(vector_results, bm25_results, method='rrf'):
        """
        Объединение результатов от разных ретриверов
        
        Args:
            vector_results: результаты векторного поиска
            bm25_results: результаты BM25 поиска
            method: метод объединения ('rrf', 'concat')
            
        Returns:
            Объединенный список документов
        """
        if method == 'rrf':
            return HybridRetriever.reciprocal_rank_fusion([vector_results, bm25_results])
        elif method == 'concat':
            return list(dict.fromkeys(vector_results + bm25_results))
        else:
            raise ValueError(f'Unknown method: {method}')
    
    def create_ensemble_with_reranking(self):
        """
        Создание ансамбля ретриверов с переранжированием
        
        Returns:
            Компрессионный ретривер с переранжированием
        """
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],

        )
        
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker(model=model, top_n=self.top_n)
        
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_retriever
        )
    
    def search(self, query, use_reranker=True):
        """
        Выполнение гибридного поиска
        
        Args:
            query: поисковый запрос
            use_reranker: использовать ли переранжирование
            
        Returns:
            Список релевантных документов
        """
        if use_reranker:
            compression_retriever = self.create_ensemble_with_reranking()
            return compression_retriever.invoke(query,)
        else:
            vector_results = self.vector_retriever.invoke(query)
            bm25_results = self.bm25_retriever.invoke(query)
            return self.merge_retriever_results(vector_results, bm25_results, method='rrf')


if __name__ == '__main__':
    # Подготовка данных
    creator = Creator()
    preprocessed_docs = Preprocessor.preprocess_documents('data')
    splitter = creator.make_splitter({'unit': 'chars', 'chunk_size': 200, 'chunk_overlap': 20})
    chunks = creator.get_chunks(preprocessed_docs, splitter)

    # Инициализация ретривера
    retriever = HybridRetriever(chunks)
    
    # Выполнение поиска
    query = 'Какова роль плана-графика потребления в системе прогнозирования и как он используется для улучшения модели?'
    results = retriever.search(query, use_reranker=True)
    print(results)
    
    # Вывод результатов
    print(f'Результаты поиска для запроса: "{query}"\n')
    for i, doc in enumerate(results, 1):
        print(f'{i}. {doc.page_content}...')
        print(f'Источник: {doc.metadata.get("source", "unknown")}')
        print()