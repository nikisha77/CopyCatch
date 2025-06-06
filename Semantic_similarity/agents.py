import time
import re
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI  # For type hinting

from .models import (
    ResearchPaperAnalysis,
    PaperSimilarityResult,
    AnalysisReport,
    ResearchPaperSection,  # ResearchPaperSection is defined but not used by agents in provided code
)
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


# Enums from original file (if needed by agents directly, or handled by orchestrator)
class TaskStatus:  # Simplified for this context
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentAction:  # Simplified
    EXTRACT_PDF = "extract_pdf"
    ANALYZE_PAPER = "analyze_paper"
    # ... other actions


class Task:  # Simplified
    def __init__(self, id: str, action: str, input_data: Dict[str, Any]):
        self.id = id
        self.action = action
        self.input_data = input_data
        self.status: str = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.completed_at: Optional[datetime] = None


class BaseAgent(ABC):
    def __init__(self, name: str, llm: ChatOpenAI):  # Expect initialized LLM
        self.name = name
        self.llm = llm
        self.task_history: List[Task] = []  # Kept for consistency

    @abstractmethod
    def execute_task(self, task: Task) -> Any:
        pass

    def log_task(self, task: Task):
        self.task_history.append(task)
        logger.info(f"[{self.name}] - Task {task.id} ({task.action}): {task.status}")

    def _api_call_with_retry(
        self,
        prompt: str,
        parser: PydanticOutputParser,
        max_retries: int = 3,
        initial_delay: int = 1,
    ) -> Any:
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=f"You are an expert {self.name} specializing in academic literature."
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                content = response.content
                # Robust JSON extraction
                json_match = re.search(
                    r"```json\s*(\{[\s\S]*?\})\s*```", content, re.DOTALL
                )
                if not json_match:
                    json_match = re.search(
                        r"(\{[\s\S]*\})", content, re.DOTALL
                    )  # Find first valid JSON block

                if json_match:
                    json_str = json_match.group(1)
                    return parser.parse(json_str)
                else:
                    # Try to parse directly if no markdown or clear block, assuming entire content might be JSON
                    try:
                        return parser.parse(content)
                    except Exception as direct_parse_err:
                        logger.error(
                            f"Direct parsing failed after no JSON block found: {direct_parse_err}"
                        )
                        raise ValueError(
                            f"No structured JSON data found in LLM response for {self.name}. Response: {content[:500]}..."
                        )

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {self.name}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2**attempt))
                else:
                    logger.error(
                        f"API call failed after {max_retries} retries for {self.name}."
                    )
                    raise


class PDFExtractionAgent(BaseAgent):
    def __init__(
        self, llm: ChatOpenAI
    ):  # llm might not be strictly needed if not using it for refinement
        super().__init__("PDFExtractionAgent", llm)
        self.pdf_processor = PDFProcessor()

    def execute_task(self, task: Task) -> str:
        task.status = TaskStatus.IN_PROGRESS
        pdf_path = ""  # Initialize for robust error message
        try:
            pdf_path = task.input_data.get("pdf_path")
            if not pdf_path:  # os.path.exists is checked by PDFProcessor
                raise ValueError("PDF path not provided in task input_data.")

            text = self.pdf_processor.extract_text(pdf_path)
            # Cleaning is now part of PDFProcessor.extract_text
            task.result = text
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(
                f"Successfully extracted {len(text)} characters from {pdf_path}"
            )
            return text
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"PDF extraction failed for {pdf_path or 'unknown path'}: {e}")
            raise
        finally:
            self.log_task(task)


class PaperAnalysisAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__("PaperAnalysisAgent", llm)
        self.parser = PydanticOutputParser(pydantic_object=ResearchPaperAnalysis)

    def execute_task(self, task: Task) -> ResearchPaperAnalysis:
        task.status = TaskStatus.IN_PROGRESS
        paper_id = "unknown_paper"
        try:
            paper_text = task.input_data.get("paper_text")
            paper_id = task.input_data.get("paper_id", "unknown_paper")
            if not paper_text:
                raise ValueError("No paper text provided for analysis.")

            prompt = self._create_analysis_prompt(paper_text)
            analysis = self._api_call_with_retry(prompt, self.parser)
            task.result = analysis
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(
                f"Successfully analyzed paper {paper_id}: {analysis.title if analysis else 'N/A'}"
            )
            return analysis
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Paper analysis failed for {paper_id}: {e}")
            raise
        finally:
            self.log_task(task)

    def _create_analysis_prompt(self, paper_text: str) -> str:
        # Truncate carefully. Max 30k chars ~ 7.5k tokens. GPT-4o-mini has 128k context.
        # This truncation might be aggressive for gpt-4o-mini. Consider model limits.
        # For now, keeping original truncation.
        truncated_text = paper_text[: min(len(paper_text), 30000)]
        return f"""You are an expert academic researcher analyzing scientific literature. Your goal is to extract structured information from a research paper.

Analyze the following research paper text and provide a comprehensive analysis structured as a JSON object, adhering strictly to the provided Pydantic schema.

Paper Text:
{truncated_text}

{self.parser.get_format_instructions()}

Ensure your output is a single JSON object. Focus on:
1. Identifying the core research question and objectives.
2. Understanding the methodology and experimental design.
3. Extracting key findings and their significance.
4. Identifying main contributions to the field.
5. Recognizing limitations and future work suggestions.
6. Categorizing technical domains and core concepts.
"""


class ComparisonAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__("ComparisonAgent", llm)
        self.parser = PydanticOutputParser(pydantic_object=PaperSimilarityResult)

    def execute_task(self, task: Task) -> PaperSimilarityResult:
        task.status = TaskStatus.IN_PROGRESS
        comparison_id = "unknown_comparison"
        try:
            paper1_analysis = task.input_data.get("paper1_analysis")
            paper2_analysis = task.input_data.get("paper2_analysis")
            comparison_id = task.input_data.get("comparison_id", "unknown_comparison")

            if not paper1_analysis or not paper2_analysis:
                raise ValueError(
                    "Both paper analyses (ResearchPaperAnalysis instances) required for comparison."
                )
            if not isinstance(paper1_analysis, ResearchPaperAnalysis) or not isinstance(
                paper2_analysis, ResearchPaperAnalysis
            ):
                raise TypeError(
                    "Input analyses must be ResearchPaperAnalysis instances."
                )

            prompt = self._create_comparison_prompt(paper1_analysis, paper2_analysis)
            comparison = self._api_call_with_retry(prompt, self.parser)
            task.result = comparison
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(
                f"Successfully compared papers {comparison_id}: similarity score {comparison.final_similarity_score if comparison else 'N/A':.2f}"
            )
            return comparison
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Paper comparison failed for {comparison_id}: {e}")
            raise
        finally:
            self.log_task(task)

    def _create_comparison_prompt(
        self, paper1: ResearchPaperAnalysis, paper2: ResearchPaperAnalysis
    ) -> str:
        return f"""You are an expert academic researcher specializing in semantic similarity analysis between research papers. Your task is to compare two provided research paper analyses and output a structured JSON object representing their similarity, adhering strictly to the provided Pydantic schema.

Paper 1 Analysis Summary:
- Title: {paper1.title}
- Research Question: {paper1.primary_research_question}
- Methodology: {paper1.methodology_summary}
- Key Findings (Top 3): {', '.join(paper1.key_findings[:3]) if paper1.key_findings else 'N/A'}
- Technical Domain: {', '.join(paper1.technical_domain) if paper1.technical_domain else 'N/A'}
- Core Concepts (Top 5): {', '.join(paper1.core_concepts[:5]) if paper1.core_concepts else 'N/A'}

Paper 2 Analysis Summary:
- Title: {paper2.title}
- Research Question: {paper2.primary_research_question}
- Methodology: {paper2.methodology_summary}
- Key Findings (Top 3): {', '.join(paper2.key_findings[:3]) if paper2.key_findings else 'N/A'}
- Technical Domain: {', '.join(paper2.technical_domain) if paper2.technical_domain else 'N/A'}
- Core Concepts (Top 5): {', '.join(paper2.core_concepts[:5]) if paper2.core_concepts else 'N/A'}

{self.parser.get_format_instructions()}

Analyze the semantic similarity between these papers, providing scores from 0 to 1 (or 0 to 5 for overall) and concise reasoning. Consider:
1. Research question alignment and objectives.
2. Methodological approaches and techniques.
3. Key findings and conclusions.
4. Technical domain overlap.
5. Conceptual frameworks and terminology.
6. Potential citation network overlap (infer from shared concepts/domains if explicit citations are not available).

Ensure your output is a single JSON object.
"""


class ReportGenerationAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__("ReportGenerationAgent", llm)
        self.parser = PydanticOutputParser(
            pydantic_object=AnalysisReport
        )  # For insights prompt

    def execute_task(self, task: Task) -> AnalysisReport:
        task.status = TaskStatus.IN_PROGRESS
        try:
            data = task.input_data
            target_analysis = data.get("target_analysis")
            comparison_analyses = data.get("comparison_analyses", [])
            similarity_results = data.get("similarity_results", [])

            if not target_analysis or not comparison_analyses or not similarity_results:
                raise ValueError(
                    "Missing data for report generation: target_analysis, comparison_analyses, or similarity_results."
                )
            if not isinstance(target_analysis, ResearchPaperAnalysis):
                raise TypeError(
                    "target_analysis must be a ResearchPaperAnalysis instance."
                )
            if not all(
                isinstance(ca, ResearchPaperAnalysis) for ca in comparison_analyses
            ):
                raise TypeError(
                    "All items in comparison_analyses must be ResearchPaperAnalysis instances."
                )
            if not all(
                isinstance(sr, PaperSimilarityResult) for sr in similarity_results
            ):
                raise TypeError(
                    "All items in similarity_results must be PaperSimilarityResult instances."
                )

            report = self._create_comprehensive_report(
                target_analysis, comparison_analyses, similarity_results
            )
            task.result = report
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info("Successfully generated comprehensive analysis report.")
            return report
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Report generation failed: {e}")
            raise
        finally:
            self.log_task(task)

    def _create_comprehensive_report(
        self,
        target_analysis: ResearchPaperAnalysis,
        comparison_analyses: List[ResearchPaperAnalysis],
        similarity_results: List[PaperSimilarityResult],
    ) -> AnalysisReport:
        generated_insights = []
        try:
            insights_prompt = self._create_insights_prompt(
                target_analysis, comparison_analyses, similarity_results
            )
            # Use self.parser which is PydanticOutputParser(pydantic_object=AnalysisReport)
            # The LLM is expected to return a JSON adhering to AnalysisReport, from which we extract key_insights
            parsed_insights_report = self._api_call_with_retry(
                prompt=insights_prompt, parser=self.parser
            )
            generated_insights = (
                parsed_insights_report.key_insights if parsed_insights_report else []
            )
        except Exception as e:
            logger.warning(
                f"LLM-based insight generation failed: {e}. Falling back to programmatic insights."
            )
            generated_insights = self._extract_key_insights_programmatically(
                similarity_results, comparison_analyses, target_analysis
            )

        if not generated_insights:  # Double fallback
            generated_insights = self._extract_key_insights_programmatically(
                similarity_results, comparison_analyses, target_analysis
            )

        summary_prompt = self._create_summary_prompt(
            target_analysis, comparison_analyses, similarity_results, generated_insights
        )
        # For summary, a plain text response is fine, no Pydantic parsing needed here.
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="You are an expert report writer, generating concise executive summaries for academic research."
                    ),
                    HumanMessage(content=summary_prompt),
                ]
            )
            generated_summary = response.content.strip()
        except Exception as e:
            logger.warning(
                f"LLM-based summary generation failed: {e}. Using a template summary."
            )
            generated_summary = f"Summary of analysis for '{target_analysis.title}'. {len(similarity_results)} comparisons made. Key insights: {'; '.join(generated_insights[:2]) if generated_insights else 'N/A'}"

        return AnalysisReport(
            summary=generated_summary,
            methodology_overview="Multi-agent system employing LLM-based semantic analysis, PDF text extraction, and structured data comparison to assess research paper similarity.",
            key_insights=generated_insights,
        )

    def _create_insights_prompt(
        self, target_analysis, comparison_analyses, similarity_results
    ) -> str:
        # Ensure comparison_analyses and similarity_results are of the same length
        num_comparisons = min(len(comparison_analyses), len(similarity_results))

        analysis_summaries = "\n".join(
            [
                f"- {comparison_analyses[i].title}: {comparison_analyses[i].primary_research_question}, Domains: {', '.join(comparison_analyses[i].technical_domain if comparison_analyses[i].technical_domain else ['N/A'])}"
                for i in range(num_comparisons)
            ]
        )
        similarity_details = "\n".join(
            [
                f"- Compared to '{comparison_analyses[i].title}': Score={similarity_results[i].final_similarity_score:.2f}, Reasoning: {similarity_results[i].reasoning}"
                for i in range(num_comparisons)
            ]
        )

        return f"""You are an expert academic analyst. Based on the following research paper analyses and their similarity results, generate a list of concise and actionable key insights. Focus on patterns, notable similarities or differences, and overall implications.

Target Paper:
Title: {target_analysis.title}
Research Question: {target_analysis.primary_research_question}
Key Findings: {', '.join(target_analysis.key_findings[:3] if target_analysis.key_findings else ['N/A'])}

Comparison Papers Summaries:
{analysis_summaries}

Similarity Results (Target vs. Comparison Papers):
{similarity_details}

{self.parser.get_format_instructions()} 

Focus ONLY on the 'key_insights' field. Provide a list of strings for key_insights. The other fields like 'summary' and 'methodology_overview' in the schema are not needed for this specific insight generation task, you can use placeholder text or omit them if the parser allows.
Generate key insights that highlight the most important observations from this analysis.
"""

    def _create_summary_prompt(
        self, target_analysis, comparison_analyses, similarity_results, insights
    ) -> str:
        min_score = (
            min(r.final_similarity_score for r in similarity_results)
            if similarity_results
            else 0
        )
        max_score = (
            max(r.final_similarity_score for r in similarity_results)
            if similarity_results
            else 0
        )

        summary_intro = f"This report provides a comprehensive similarity analysis of several research papers against the target paper '{target_analysis.title}' (Research Question: '{target_analysis.primary_research_question}').\n\n"
        similarity_overview = f"Overall similarity scores range from {min_score:.2f} to {max_score:.2f} (out of 5), indicating varying degrees of thematic and methodological overlap."
        insights_section = (
            "Key insights derived from this analysis include:\n"
            + "\n".join([f"- {i}" for i in insights])
            if insights
            else ""
        )

        return f"""Generate a concise executive summary for a report on research paper similarity analysis.

{summary_intro}
{similarity_overview}

Consider the following key insights from the analysis:
{insights_section if insights else "No specific insights were provided."}

The summary should cover the purpose of the analysis, the overall findings regarding similarity, and the most important insights. Output only the summary text.
"""

    def _extract_key_insights_programmatically(
        self,
        similarity_results: List[PaperSimilarityResult],
        analyses: List[ResearchPaperAnalysis],  # comparison_analyses
        target_analysis: ResearchPaperAnalysis,
    ) -> List[str]:
        insights = []
        if not similarity_results:
            return ["No similarity results available to generate insights."]

        scores = [r.final_similarity_score for r in similarity_results]
        if scores:
            avg_score = np.mean(scores)
            insights.append(
                f"Average similarity score of '{target_analysis.title}' against {len(analyses)} papers: {avg_score:.2f} (out of 5)."
            )

            high_sim_papers = [
                (analyses[i].title, s.final_similarity_score)
                for i, s in enumerate(similarity_results)
                if s.final_similarity_score >= 4.0
            ]
            if high_sim_papers:
                insights.append(
                    f"High similarity (>=4.0/5) with: {', '.join([f'{title} ({score:.2f})' for title, score in high_sim_papers])}."
                )

            low_sim_papers = [
                (analyses[i].title, s.final_similarity_score)
                for i, s in enumerate(similarity_results)
                if s.final_similarity_score <= 2.0
            ]
            if low_sim_papers:
                insights.append(
                    f"Low similarity (<=2.0/5) with: {', '.join([f'{title} ({score:.2f})' for title, score in low_sim_papers])}, suggesting divergent research."
                )

        all_paper_analyses = [target_analysis] + analyses
        all_domains = set(
            d
            for paper in all_paper_analyses
            if paper.technical_domain
            for d in paper.technical_domain
        )
        all_concepts = set(
            c
            for paper in all_paper_analyses
            if paper.core_concepts
            for c in paper.core_concepts
        )

        if len(all_domains) > 0:
            insights.append(
                f"Collective research spans domains like: {', '.join(list(all_domains)[:3])}{'...' if len(all_domains) > 3 else '.'}"
            )
        if len(all_concepts) > 0:
            insights.append(
                f"Key shared concepts include: {', '.join(list(all_concepts)[:3])}{'...' if len(all_concepts) > 3 else '.'}"
            )

        return (
            insights
            if insights
            else [
                "Basic analysis complete. No specific programmatic insights generated beyond scores."
            ]
        )
