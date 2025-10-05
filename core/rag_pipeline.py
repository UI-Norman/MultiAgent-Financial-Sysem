from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import json
import os
from openai import OpenAI

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict
    score: float
    source: str  # "dense", "sparse", "hybrid"

class AdvancedRAGPipeline:
    """
    Multi-stage retrieval:
    1. Query decomposition
    2. Hybrid search (dense + sparse)
    3. Re-ranking with cross-encoder
    4. Cross-document comparison
    """
    
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents if documents else []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Sparse retrieval (BM25) - initialize only if documents available
        if self.documents:
            tokenized_docs = [doc.content.split() if hasattr(doc, 'content') else str(doc).split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None
        
        # Re-ranker - initialize lazily to avoid loading model if not needed
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy-load the reranker model"""
        if self._reranker is None:
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self._reranker
    
    def decompose_query(self, query: str) -> List[str]:
        """Break complex query into sub-queries using LLM"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a query decomposition expert. Break complex queries into 2-3 focused sub-queries. Return ONLY a valid JSON array of strings."},
                    {"role": "user", "content": f"Break this query into 2-3 focused sub-queries:\n\nQuery: {query}\n\nReturn as JSON array: [\"sub-query 1\", \"sub-query 2\"]"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            # Try to parse as JSON
            try:
                sub_queries = json.loads(result)
                if isinstance(sub_queries, list) and sub_queries:
                    return sub_queries
            except json.JSONDecodeError:
                pass
            
            # Fallback: return original query
            return [query]
            
        except Exception as e:
            # Fallback: if LLM fails, return original query
            return [query]
    
    def hybrid_search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Combine dense + sparse retrieval"""
        
        # Dense retrieval (vector similarity)
        dense_results = self.vector_store.similarity_search(query, k=k) if self.vector_store else []
        
        # Sparse retrieval (BM25) - only if documents available
        sparse_results = []
        if self.bm25 and self.documents:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            sparse_indices = np.argsort(bm25_scores)[-k:][::-1]
            sparse_results = [self.documents[i] for i in sparse_indices]
        
        # Merge and deduplicate
        combined = self._merge_results(dense_results, sparse_results)
        return combined
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank using cross-encoder"""
        pairs = [[query, r.content] for r in results]
        scores = self.reranker.predict(pairs)
        
        for result, score in zip(results, scores):
            result.score = score
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def compare_across_years(self, query: str, years: List[str]) -> Dict[str, List[RetrievalResult]]:
        """Retrieve same topic across multiple years"""
        results_by_year = {}
        
        for year in years:
            # Add year filter to metadata
            filtered_query = f"{query} (year: {year})"
            results = self.hybrid_search(filtered_query, k=5)
            results_by_year[year] = results
        
        return results_by_year
    
    def retrieve(self, query: str, strategy: str = "hybrid") -> List[RetrievalResult]:
        """Main retrieval entry point"""
        
        # Step 1: Decompose if complex
        sub_queries = self.decompose_query(query)
        
        all_results = []
        for sq in sub_queries:
            # Step 2: Hybrid search
            results = self.hybrid_search(sq, k=20)
            
            # Step 3: Re-rank
            if results:  # Only rerank if we have results
                results = self.rerank(sq, results)
            
            all_results.extend(results[:5])  # Top 5 per sub-query
        
        # Step 4: Final deduplication
        unique_results = self._deduplicate(all_results)
        
        return unique_results[:10]  # Return top 10 overall
    
    def _merge_results(self, dense_results: List, sparse_results: List) -> List[RetrievalResult]:
        """Merge dense and sparse retrieval results using RRF (Reciprocal Rank Fusion)"""
        results_dict = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            content = result.page_content if hasattr(result, 'page_content') else str(result)
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            
            if content not in results_dict:
                results_dict[content] = {
                    'content': content,
                    'metadata': metadata,
                    'score': 0,
                    'source': 'dense'
                }
            
            # RRF score: 1 / (k + rank), k=60 is standard
            results_dict[content]['score'] += 1 / (60 + rank)
            results_dict[content]['source'] = 'hybrid'
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            content = result.content if hasattr(result, 'content') else str(result)
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            
            if content not in results_dict:
                results_dict[content] = {
                    'content': content,
                    'metadata': metadata,
                    'score': 0,
                    'source': 'sparse'
                }
            
            results_dict[content]['score'] += 1 / (60 + rank)
            if results_dict[content]['source'] == 'dense':
                results_dict[content]['source'] = 'hybrid'
        
        # Convert to RetrievalResult objects and sort by score
        merged = [
            RetrievalResult(
                content=r['content'],
                metadata=r['metadata'],
                score=r['score'],
                source=r['source']
            )
            for r in results_dict.values()
        ]
        
        return sorted(merged, key=lambda x: x.score, reverse=True)
    
    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on content similarity"""
        if not results:
            return []
        
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Simple deduplication based on exact content match
            content_key = result.content[:200]  # Use first 200 chars as key
            
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_results.append(result)
        
        return unique_results