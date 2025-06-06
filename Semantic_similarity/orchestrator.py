import datetime
import os
import time
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm # For console progress, not Streamlit

from langchain_openai import ChatOpenAI, OpenAIEmbeddings # For type hinting

from .agents import (
    PDFExtractionAgent, PaperAnalysisAgent, ComparisonAgent, ReportGenerationAgent, Task, AgentAction
)
from .models import ResearchPaperAnalysis, PaperSimilarityResult, AnalysisReport

logger = logging.getLogger(__name__)
MAX_WORKERS_SEMANTIC = 5 # Default from original, can be configured

class AgenticResearchPaperAnalyzer:
    def __init__(self, llm_client: ChatOpenAI, embeddings_model_client: OpenAIEmbeddings, max_workers: int = MAX_WORKERS_SEMANTIC):
        self.llm = llm_client
        self.embeddings_model = embeddings_model_client # Stored, though not directly used by orchestrator in provided snippet

        self.pdf_agent = PDFExtractionAgent(self.llm)
        self.analysis_agent = PaperAnalysisAgent(self.llm)
        self.comparison_agent = ComparisonAgent(self.llm)
        self.report_agent = ReportGenerationAgent(self.llm)

        self.task_counter = 0
        # self.results_cache = {} # Original had this, but not used.
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _generate_task_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter}_{int(time.time())}"

    def _get_paper_title_from_analysis(self, analysis: ResearchPaperAnalysis) -> str:
        # Helper to get title for mapping, assuming analysis object is available
        return analysis.title if analysis and analysis.title else "Untitled Paper"

    def analyze_papers_from_pdfs(self, target_pdf_path: str, comparison_pdf_paths: List[str],
                                output_dir: str = "analysis_results_semantic") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting agentic analysis of {target_pdf_path} against {len(comparison_pdf_paths)} papers.")
        
        all_results = {
            "report": None, "target_analysis": None, 
            "comparison_analyses": [None] * len(comparison_pdf_paths), 
            "similarity_results": [None] * len(comparison_pdf_paths)
        }

        try:
            # --- Step 1: Extract text from all PDFs in parallel ---
            logger.info("Step 1: Extracting text from PDFs...")
            pdf_extraction_futures = {}
            
            # Target PDF
            pdf_extraction_futures[self.executor.submit(self._extract_pdf_text, target_pdf_path, "target")] = ("target", 0)
            # Comparison PDFs
            for i, comp_pdf_path in enumerate(comparison_pdf_paths):
                pdf_extraction_futures[self.executor.submit(self._extract_pdf_text, comp_pdf_path, f"comparison_{i}")] = ("comparison", i)

            extracted_texts = {"target": None, "comparison": [None] * len(comparison_pdf_paths)}
            for future in tqdm(as_completed(pdf_extraction_futures), total=len(pdf_extraction_futures), desc="Extracting PDF Texts"):
                text_type, index = pdf_extraction_futures[future]
                try:
                    text_content = future.result()
                    if text_type == "target":
                        extracted_texts["target"] = text_content
                    else: # comparison
                        extracted_texts["comparison"][index] = text_content
                except Exception as e:
                    logger.error(f"Failed text extraction for {text_type} {index if text_type=='comparison' else ''}: {e}")
                    # Decide if to re-raise or mark as failed and continue
                    if text_type == "target": raise Exception(f"Target PDF extraction failed: {e}") from e # Critical
                    # For comparison, we might allow some to fail
                    extracted_texts["comparison"][index] = f"EXTRACTION_FAILED: {e}"


            if not extracted_texts["target"]:
                raise ValueError("Target PDF text extraction failed. Cannot proceed.")

            # --- Step 2: Analyze paper content in parallel ---
            logger.info("Step 2: Analyzing paper content...")
            paper_analysis_futures = {}
            # Target Analysis
            paper_analysis_futures[self.executor.submit(self._analyze_paper, extracted_texts["target"], "target")] = ("target", 0)
            # Comparison Analyses
            for i, comp_text in enumerate(extracted_texts["comparison"]):
                if isinstance(comp_text, str) and not comp_text.startswith("EXTRACTION_FAILED"):
                     paper_analysis_futures[self.executor.submit(self._analyze_paper, comp_text, f"comparison_{i}")] = ("comparison", i)
                else: # Skip analysis if extraction failed
                    all_results["comparison_analyses"][i] = f"ANALYSIS_SKIPPED_DUE_TO_EXTRACTION_FAILURE: {comp_text}"


            for future in tqdm(as_completed(paper_analysis_futures), total=len(paper_analysis_futures), desc="Analyzing Papers"):
                analysis_type, index = paper_analysis_futures[future]
                try:
                    analysis_result = future.result()
                    if analysis_type == "target":
                        all_results["target_analysis"] = analysis_result
                    else: # comparison
                        all_results["comparison_analyses"][index] = analysis_result
                except Exception as e:
                    logger.error(f"Failed paper analysis for {analysis_type} {index if analysis_type=='comparison' else ''}: {e}")
                    if analysis_type == "target": raise Exception(f"Target paper analysis failed: {e}") from e
                    all_results["comparison_analyses"][index] = f"ANALYSIS_FAILED: {e}"


            if not all_results["target_analysis"] or not isinstance(all_results["target_analysis"], ResearchPaperAnalysis) :
                raise ValueError("Target paper analysis failed or yielded invalid result. Cannot proceed.")

            # --- Step 3: Compare target paper with each comparison paper in parallel ---
            logger.info("Step 3: Comparing papers...")
            comparison_futures = {}
            for i, comp_analysis in enumerate(all_results["comparison_analyses"]):
                if isinstance(comp_analysis, ResearchPaperAnalysis): # Only compare if analysis was successful
                    comparison_futures[
                        self.executor.submit(self._compare_papers, all_results["target_analysis"], comp_analysis, f"target_vs_comp_{i}")
                    ] = i # Store index to map result

            for future in tqdm(as_completed(comparison_futures), total=len(comparison_futures), desc="Comparing Papers"):
                comp_index = comparison_futures[future]
                try:
                    similarity_result = future.result()
                    all_results["similarity_results"][comp_index] = similarity_result
                except Exception as e:
                    logger.error(f"Failed comparison for comparison paper index {comp_index}: {e}")
                    all_results["similarity_results"][comp_index] = f"COMPARISON_FAILED: {e}"
            
            # Filter out failed comparisons for report generation
            valid_comparison_analyses = [ca for ca in all_results["comparison_analyses"] if isinstance(ca, ResearchPaperAnalysis)]
            valid_similarity_results = [sr for sr in all_results["similarity_results"] if isinstance(sr, PaperSimilarityResult)]


            # --- Step 4: Generate comprehensive report ---
            logger.info("Step 4: Generating comprehensive report...")
            if valid_comparison_analyses and valid_similarity_results: # Ensure there's something to report on
                all_results["report"] = self._generate_report(
                    all_results["target_analysis"], 
                    valid_comparison_analyses, # Use only successfully analyzed ones
                    valid_similarity_results   # Use only successfully compared ones
                )
            else:
                logger.warning("Not enough valid comparison data to generate a full report. Report will be minimal.")
                all_results["report"] = AnalysisReport(
                    summary=f"Analysis for '{all_results['target_analysis'].title}' completed with issues. Limited comparison data available.",
                    methodology_overview="Standard multi-agent analysis attempted.",
                    key_insights=["Report generation limited due to failures in prior steps."]
                )


            self._save_results_to_files(all_results, output_dir)
            logger.info("Agentic analysis completed!")
            return all_results

        except Exception as e:
            logger.error(f"Overall agentic analysis pipeline failed: {e}", exc_info=True)
            # Save whatever partial results might exist
            self._save_results_to_files(all_results, output_dir, error_suffix="_ERROR")
            raise # Re-raise the exception to be caught by the UI layer
        finally:
            self.executor.shutdown(wait=False) # Allow main thread to exit, don't wait for all tasks if error

    def _extract_pdf_text(self, pdf_path: str, paper_id: str) -> str:
        task = Task(id=self._generate_task_id(), action=AgentAction.EXTRACT_PDF, 
                    input_data={"pdf_path": pdf_path, "paper_id": paper_id})
        return self.pdf_agent.execute_task(task)

    def _analyze_paper(self, paper_text: str, paper_id: str) -> ResearchPaperAnalysis:
        task = Task(id=self._generate_task_id(), action=AgentAction.ANALYZE_PAPER, 
                    input_data={"paper_text": paper_text, "paper_id": paper_id})
        return self.analysis_agent.execute_task(task)

    def _compare_papers(self, paper1_analysis: ResearchPaperAnalysis, 
                       paper2_analysis: ResearchPaperAnalysis, comparison_id: str) -> PaperSimilarityResult:
        task = Task(id=self._generate_task_id(), action="compare_papers", # Using string for action
                    input_data={"paper1_analysis": paper1_analysis, 
                                "paper2_analysis": paper2_analysis, 
                                "comparison_id": comparison_id})
        return self.comparison_agent.execute_task(task)

    def _generate_report(self, target_analysis, comparison_analyses, similarity_results) -> AnalysisReport:
        task = Task(id=self._generate_task_id(), action="generate_report",
                    input_data={"target_analysis": target_analysis, 
                                "comparison_analyses": comparison_analyses, 
                                "similarity_results": similarity_results})
        return self.report_agent.execute_task(task)

    def _save_results_to_files(self, results_dict: Dict[str, Any], output_dir: str, error_suffix=""):
        # Save main report if it exists
        if results_dict.get("report") and isinstance(results_dict["report"], AnalysisReport):
            report_path = os.path.join(output_dir, f"analysis_report{error_suffix}.json")
            summary_path = os.path.join(output_dir, f"analysis_summary{error_suffix}.txt")
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(results_dict["report"].model_dump(), f, indent=2)
                logger.info(f"Full JSON report saved to {report_path}")

                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("--- Executive Summary ---\n")
                    f.write(results_dict["report"].summary + "\n\n")
                    f.write("--- Methodology Overview ---\n")
                    f.write(results_dict["report"].methodology_overview + "\n\n")
                    f.write("--- Key Insights ---\n")
                    for insight in results_dict["report"].key_insights:
                        f.write(f"- {insight}\n")
                logger.info(f"Summary text report saved to {summary_path}")
            except Exception as e:
                logger.error(f"Error saving report files: {e}")
        else:
            logger.warning("Report object not found or invalid in results, skipping report file saving.")

        # Optionally, save all raw analysis data too
        all_data_path = os.path.join(output_dir, f"all_analysis_data{error_suffix}.json")
        try:
            # Need a custom serializer for Pydantic models if not using .model_dump() on each
            def pydantic_dumper(obj):
                if hasattr(obj, 'model_dump_json'):
                    return json.loads(obj.model_dump_json()) # Creates dict from json string
                if isinstance(obj, (datetime,)):
                     return obj.isoformat()
                return str(obj) # Fallback

            with open(all_data_path, 'w', encoding='utf-8') as f:
                # Create a serializable version of results_dict
                serializable_results = {}
                for k, v in results_dict.items():
                    if k == "report" and isinstance(v, AnalysisReport):
                        serializable_results[k] = v.model_dump()
                    elif k == "target_analysis" and isinstance(v, ResearchPaperAnalysis):
                        serializable_results[k] = v.model_dump()
                    elif k == "comparison_analyses":
                        serializable_results[k] = [item.model_dump() if isinstance(item, ResearchPaperAnalysis) else str(item) for item in v]
                    elif k == "similarity_results":
                        serializable_results[k] = [item.model_dump() if isinstance(item, PaperSimilarityResult) else str(item) for item in v]
                    else:
                        serializable_results[k] = v
                json.dump(serializable_results, f, indent=2) # default=pydantic_dumper - model_dump is better
            logger.info(f"All analysis data saved to {all_data_path}")
        except Exception as e:
            logger.error(f"Error saving all analysis data: {e}")