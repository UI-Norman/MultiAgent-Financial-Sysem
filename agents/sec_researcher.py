from phi.agent import Agent
from phi.knowledge.agent import AgentKnowledge
from core.rag_pipeline import AdvancedRAGPipeline
from core.citations import CitationTracker, Citation
from typing import List, Dict
from phi.model.openai import OpenAIChat


class SECResearcherAgent:
    """Advanced RAG agent for 10-K analysis"""
    
    def __init__(self, knowledge_base, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.citation_tracker = CitationTracker()
        self.knowledge_base = knowledge_base
        
        # Create agent without knowledge_base parameter to avoid type mismatch
        # We'll use our custom RAG pipeline instead
        self.agent = Agent(
            name="SEC Researcher",
            model=OpenAIChat(id="gpt-4o"),
            role="Extract insights from 10-K filings",
            instructions=[
                "Use advanced retrieval strategies from provided context",
                "Compare trends across 5 years",
                "Always cite: [Ticker 10-K Year, Section]",
                "Return structured evidence with provenance"
            ]
        )

    # Add missing method:
    def _parse_financial_data(self, results: List) -> Dict:
        """Extract structured financial data from results"""
        # Simplified implementation
        return {
            "revenue_trend": "N/A - requires 10-K indexing",
            "profit_margin": "N/A",
            "cash_flow": "N/A"
        }

    # Add run method that orchestrator calls:
    def run(self, task: str, ticker: str) -> Dict:
        """Execute SEC research task"""
        # For now, return placeholder
        return {
            "risks": {},
            "financials": {},
            "strategy": "See 10-K filings for details"
        }
  
    def analyze_risks(self, ticker: str, years: List[str]) -> Dict:
        """Analyze risk factors across multiple years"""
        
        query = f"What are the key business risks for {ticker}?"
        
        # Cross-year comparison
        results_by_year = self.rag_pipeline.compare_across_years(query, years)
        
        risk_analysis = {}
        for year, results in results_by_year.items():
            risks = []
            for result in results:
                risks.append({
                    "risk": result.content,
                    "citation": Citation(
                        source_type="10-K",
                        ticker=ticker,
                        year=year,
                        section=result.metadata.get('section'),
                        url=result.metadata.get('source_url')
                    )
                })
                
            risk_analysis[year] = risks
        
        return risk_analysis
    
    def get_financial_trends(self, ticker: str) -> Dict:
        """Extract financial performance trends"""
        
        query = f"Revenue, profit margins, and cash flow trends for {ticker}"
        
        results = self.rag_pipeline.retrieve(query, strategy="hybrid")
        
        # Extract structured data
        trends = self._parse_financial_data(results)
        
        return trends