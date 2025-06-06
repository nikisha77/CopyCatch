from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchPaperSection(BaseModel):
    section_type: str = Field(..., description="Type of section")
    main_findings: str = Field(..., description="Primary findings or points")
    key_terminology: List[str] = Field(
        default_factory=list, description="Important technical terms"
    )
    citations_referenced: List[str] = Field(
        default_factory=list, description="Key citations"
    )


class ResearchPaperAnalysis(BaseModel):
    title: str = Field(..., description="Title of the research paper")
    primary_research_question: str = Field(..., description="Main research question")
    methodology_summary: str = Field(..., description="Summary of methodology")
    key_findings: List[str] = Field(..., description="Key findings or conclusions")
    main_contributions: List[str] = Field(..., description="Main contributions")
    limitations: List[str] = Field(..., description="Research limitations")
    future_work: List[str] = Field(..., description="Future research directions")
    technical_domain: List[str] = Field(..., description="Technical domains")
    core_concepts: List[str] = Field(..., description="Core concepts")


class PaperSimilarityResult(BaseModel):
    research_question_alignment: float = Field(
        ..., description="Research question alignment (0-1)"
    )
    methodology_alignment: float = Field(..., description="Methodology alignment (0-1)")
    findings_alignment: float = Field(..., description="Findings alignment (0-1)")
    domain_relevance: float = Field(..., description="Domain relevance (0-1)")
    conceptual_overlap: float = Field(..., description="Conceptual overlap (0-1)")
    citation_network_overlap: float = Field(..., description="Citation overlap (0-1)")
    final_similarity_score: float = Field(..., description="Overall similarity (0-5)")
    reasoning: str = Field(
        ..., description="Analysis reasoning in short and crisp manner"
    )


class AnalysisReport(BaseModel):
    summary: str = Field(..., description="Executive summary")
    methodology_overview: str = Field(..., description="Analysis methodology")
    key_insights: List[str] = Field(..., description="Key insights discovered")
