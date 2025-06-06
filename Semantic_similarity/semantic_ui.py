import streamlit as st
import os
import time
import json
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging

from .orchestrator import AgenticResearchPaperAnalyzer
from .models import (
    ResearchPaperAnalysis,
    PaperSimilarityResult,
    AnalysisReport,
)  # For type hints

logger = logging.getLogger(__name__)


# --- Helper functions from app.py (adapted) ---
def save_uploaded_file_temp(uploaded_file, prefix: str) -> str:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix=f"{prefix}_"
        ) as temp_f:
            temp_f.write(uploaded_file.getvalue())
            return temp_f.name
    return None


def display_concept_tags_semantic(items: List[str], tag_type: str = "secondary"):
    if items:
        tags_html = "".join(
            [f'<span class="tag tag-{tag_type}">{item}</span>' for item in items]
        )
        st.markdown(tags_html, unsafe_allow_html=True)


# --- UI Rendering Functions ---
def render_similarity_visualizations(
    similarity_results: List[PaperSimilarityResult],
    comparison_analyses: List[ResearchPaperAnalysis],
):
    st.markdown(
        '<h3 class="section-header">Similarity Dimensions Overview</h3>',
        unsafe_allow_html=True,
    )

    # Filter out non-PaperSimilarityResult items before processing
    valid_similarity_results = [
        r for r in similarity_results if isinstance(r, PaperSimilarityResult)
    ]
    # Ensure comparison_analyses matches valid_similarity_results length for zipping
    valid_comparison_analyses = [
        ca
        for i, ca in enumerate(comparison_analyses)
        if isinstance(similarity_results[i], PaperSimilarityResult)
    ]

    if not valid_similarity_results:
        st.info("No valid similarity results to visualize.")
        return

    radar_chart = go.Figure()
    colors = px.colors.qualitative.Plotly  # More colors

    for index, (result, analysis) in enumerate(
        zip(valid_similarity_results, valid_comparison_analyses)
    ):
        if not isinstance(result, PaperSimilarityResult) or not isinstance(
            analysis, ResearchPaperAnalysis
        ):
            continue  # Skip if data is not as expected (e.g., error string)

        radar_chart.add_trace(
            go.Scatterpolar(
                r=[
                    result.research_question_alignment,
                    result.methodology_alignment,
                    result.findings_alignment,
                    result.domain_relevance,
                    result.conceptual_overlap,
                    result.citation_network_overlap,
                ],
                theta=[
                    "Research Q",
                    "Methodology",
                    "Findings",
                    "Domain",
                    "Concepts",
                    "Citations",
                ],
                fill="toself",
                name=(
                    analysis.title[:30] + "..."
                    if len(analysis.title) > 30
                    else analysis.title
                ),
                line_color=colors[index % len(colors)],
            )
        )
    radar_chart.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=dict(
            text="Similarity Dimensions Radar Chart",
            font=dict(size=20),
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    st.plotly_chart(radar_chart, use_container_width=True)


def render_similarity_breakdown_table(
    similarity_results: List[PaperSimilarityResult],
    comparison_analyses: List[ResearchPaperAnalysis],
):
    st.markdown("<h4>Detailed Similarity Breakdown</h4>", unsafe_allow_html=True)

    similarity_data = []
    for index, result in enumerate(similarity_results):
        if (
            not isinstance(result, PaperSimilarityResult)
            or index >= len(comparison_analyses)
            or not isinstance(comparison_analyses[index], ResearchPaperAnalysis)
        ):
            # Add a row indicating failure for this comparison if needed, or just skip
            # For now, skip problematic entries
            continue

        analysis = comparison_analyses[index]
        similarity_data.append(
            {
                "Paper": analysis.title,
                "Overall Score": result.final_similarity_score,
                "Research Align.": result.research_question_alignment,
                "Methodology": result.methodology_alignment,
                "Findings": result.findings_alignment,
                "Domain Rel.": result.domain_relevance,
                "Concept Overlap": result.conceptual_overlap,
            }
        )

    if not similarity_data:
        st.info("No valid similarity data to display in table.")
        return

    df = pd.DataFrame(similarity_data)
    progress_cols = {
        "Overall Score": {"min_value": 0, "max_value": 5, "format": "%.2f"},
        "Research Align.": {"min_value": 0, "max_value": 1, "format": "%.2f"},
        "Methodology": {"min_value": 0, "max_value": 1, "format": "%.2f"},
        "Findings": {"min_value": 0, "max_value": 1, "format": "%.2f"},
        "Domain Rel.": {"min_value": 0, "max_value": 1, "format": "%.2f"},
        "Concept Overlap": {"min_value": 0, "max_value": 1, "format": "%.2f"},
    }
    column_config = {
        col: st.column_config.ProgressColumn(col, help=f"{col} score", **params)
        for col, params in progress_cols.items()
    }
    column_config["Paper"] = st.column_config.TextColumn("Paper Title")

    st.dataframe(
        df.set_index("Paper"), use_container_width=True, column_config=column_config
    )


def render_paper_analysis_details(analysis: ResearchPaperAnalysis, title_prefix: str):
    if not isinstance(analysis, ResearchPaperAnalysis):
        st.warning(f"{title_prefix}: Analysis data is not available or invalid.")
        if isinstance(analysis, str):  # If it's an error message string
            st.caption(f"Details: {analysis}")
        return

    with st.expander(f"{title_prefix}: {analysis.title}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Research Question:** {analysis.primary_research_question}")
            st.markdown(f"**Methodology:** {analysis.methodology_summary}")
            st.markdown("**Technical Domains:**")
            display_concept_tags_semantic(analysis.technical_domain or [], "secondary")
        with col2:
            st.markdown("**Key Findings:**")
            for finding in (analysis.key_findings or [])[:3]:
                st.write(f"• {finding}")
            st.markdown("**Main Contributions:**")
            for contrib in (analysis.main_contributions or [])[:3]:
                st.write(f"• {contrib}")
        st.markdown("**Core Concepts:**")
        display_concept_tags_semantic(analysis.core_concepts or [], "primary")


def render_detailed_comparison_metrics(
    similarity_results: List[PaperSimilarityResult],
    comparison_analyses: List[ResearchPaperAnalysis],
):
    st.markdown(
        '<h3 class="section-header">In-depth Comparison Metrics & Reasoning</h3>',
        unsafe_allow_html=True,
    )
    for index, result in enumerate(similarity_results):
        if (
            not isinstance(result, PaperSimilarityResult)
            or index >= len(comparison_analyses)
            or not isinstance(comparison_analyses[index], ResearchPaperAnalysis)
        ):
            # st.warning(f"Detailed analysis for comparison {index+1} is unavailable due to missing data.")
            continue  # Skip if data is not as expected

        analysis = comparison_analyses[index]
        with st.expander(f"Comparison with: {analysis.title}", expanded=False):
            st.markdown(
                f"**Overall Similarity Score: <span class='similarity-score'>{result.final_similarity_score:.2f}/5.0</span>**",
                unsafe_allow_html=True,
            )

            metrics_html = f"""
            <div class="horizontal-metrics">
                <div class="metric-item"><div class="metric-value">{result.research_question_alignment:.2f}</div><div class="metric-label">Research Q.</div></div>
                <div class="metric-item"><div class="metric-value">{result.methodology_alignment:.2f}</div><div class="metric-label">Methodology</div></div>
                <div class="metric-item"><div class="metric-value">{result.findings_alignment:.2f}</div><div class="metric-label">Findings</div></div>
                <div class="metric-item"><div class="metric-value">{result.domain_relevance:.2f}</div><div class="metric-label">Domain</div></div>
                <div class="metric-item"><div class="metric-value">{result.conceptual_overlap:.2f}</div><div class="metric-label">Concepts</div></div>
                <div class="metric-item"><div class="metric-value">{result.citation_network_overlap:.2f}</div><div class="metric-label">Citations</div></div>
            </div>"""
            st.markdown(metrics_html, unsafe_allow_html=True)
            st.markdown("**Reasoning:**")
            st.info(result.reasoning)


def render_semantic_analyzer_ui(llm_client, embeddings_model_client):
    st.header("Semantic Similarity Analyzer for Research Papers")
    st.markdown(
        "Upload a source paper and one or more comparison papers to analyze their semantic similarity based on content and structure."
    )
    st.markdown("---")

    # Initialize analyzer if not already done (e.g. if API key was entered later)
    # For this structure, analyzer is created once in main_app.py
    # Here we assume it's passed and valid.
    if (
        "semantic_analyzer_instance" not in st.session_state
        or st.session_state.semantic_analyzer_instance is None
    ):
        try:
            st.session_state.semantic_analyzer_instance = AgenticResearchPaperAnalyzer(
                llm_client=llm_client, embeddings_model_client=embeddings_model_client
            )  # Use clients passed from main_app
        except Exception as e:
            st.error(f"Failed to initialize Semantic Analyzer: {e}")
            return

    analyzer: AgenticResearchPaperAnalyzer = st.session_state.semantic_analyzer_instance

    # File Uploads
    col1, col2 = st.columns(2)
    with col1:
        source_pdf_upload = st.file_uploader(
            "Upload Source Research Paper (PDF)", type="pdf", key="semantic_source_pdf"
        )
    with col2:
        comparison_pdf_uploads = st.file_uploader(
            "Upload Comparison Papers (PDFs)",
            type="pdf",
            accept_multiple_files=True,
            key="semantic_comp_pdfs",
        )

    if st.button(
        "Start Semantic Analysis",
        type="primary",
        key="semantic_start_button",
        disabled=not (source_pdf_upload and comparison_pdf_uploads),
    ):
        st.session_state.semantic_processing_status = "processing"
        st.session_state.semantic_analysis_results = None  # Clear previous results

        source_temp_path = save_uploaded_file_temp(
            source_pdf_upload, "semantic_source_"
        )
        comparison_temp_paths = [
            save_uploaded_file_temp(f, f"semantic_comp_{i}_")
            for i, f in enumerate(comparison_pdf_uploads)
        ]

        # Filter out None paths if any upload failed
        comparison_temp_paths = [p for p in comparison_temp_paths if p]

        if not source_temp_path or not comparison_temp_paths:
            st.error("File saving failed. Please try uploading again.")
            st.session_state.semantic_processing_status = "error"
            # Clean up any successfully saved files
            if source_temp_path and os.path.exists(source_temp_path):
                os.unlink(source_temp_path)
            for p in comparison_temp_paths:
                if p and os.path.exists(p):
                    os.unlink(p)
            return

        status_placeholder = st.empty()
        try:
            status_placeholder.info(
                "🔄 Analysis in progress... This may take several minutes depending on the number and size of papers."
            )

            # This is a blocking call. For true async UI, would need more complex setup.
            # The orchestrator itself uses ThreadPoolExecutor for internal parallelism.
            results = analyzer.analyze_papers_from_pdfs(
                source_temp_path, comparison_temp_paths
            )

            st.session_state.semantic_analysis_results = results
            st.session_state.semantic_processing_status = "completed"
            status_placeholder.success("✅ Semantic analysis completed!")

        except Exception as e:
            st.session_state.semantic_processing_status = "error"
            status_placeholder.error(f"❌ Analysis failed: {e}")
            logger.error("Semantic analysis pipeline error", exc_info=True)
        finally:
            # Clean up temporary files
            if source_temp_path and os.path.exists(source_temp_path):
                os.unlink(source_temp_path)
            for p in comparison_temp_paths:
                if p and os.path.exists(p):
                    os.unlink(p)

    # Display Results
    if st.session_state.get("semantic_analysis_results"):
        results = st.session_state.semantic_analysis_results
        st.markdown("---")
        st.subheader("Semantic Analysis Results")

        # 1. Report Summary & Key Insights
        if results.get("report") and isinstance(results["report"], AnalysisReport):
            report = results["report"]
            st.markdown(f"<h4>Executive Summary</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-card'>{report.summary}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"<h4>Key Insights</h4>", unsafe_allow_html=True)
            if report.key_insights:
                for insight in report.key_insights:
                    st.markdown(
                        f"<div class='recommendation-card'>💡 {insight}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No specific key insights generated.")
            st.markdown(f"**Methodology Used:** {report.methodology_overview}")
        else:
            st.warning("Main analysis report is not available.")

        # 2. Visualizations & Tables
        if results.get("similarity_results") and results.get("comparison_analyses"):
            render_similarity_visualizations(
                results["similarity_results"], results["comparison_analyses"]
            )
            st.markdown("---")
            render_similarity_breakdown_table(
                results["similarity_results"], results["comparison_analyses"]
            )
        else:
            st.info(
                "Similarity scores or comparison analyses data missing for detailed breakdown."
            )

        st.markdown("---")
        # 3. Detailed Paper Analyses (Source and Comparisons)
        st.markdown("<h4>Individual Paper Summaries</h4>", unsafe_allow_html=True)
        if results.get("target_analysis"):
            render_paper_analysis_details(results["target_analysis"], "Source Paper")

        if results.get("comparison_analyses"):
            for i, comp_analysis in enumerate(results["comparison_analyses"]):
                render_paper_analysis_details(comp_analysis, f"Comparison Paper {i+1}")

        st.markdown("---")
        # 4. In-depth Comparison Metrics
        if results.get("similarity_results") and results.get("comparison_analyses"):
            render_detailed_comparison_metrics(
                results["similarity_results"], results["comparison_analyses"]
            )

        # 5. Download options
        st.markdown("---")
        st.markdown("<h4>Download Full Analysis</h4>", unsafe_allow_html=True)
        if results.get("report") and isinstance(results["report"], AnalysisReport):
            json_report_data = json.dumps(results["report"].model_dump(), indent=2)
            st.download_button(
                label="Download Report (JSON)",
                data=json_report_data,
                file_name=f"semantic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
            )

        # CSV data for similarities
        if results.get("similarity_results") and results.get("comparison_analyses"):
            csv_data = []
            for i, sr in enumerate(results["similarity_results"]):
                if (
                    isinstance(sr, PaperSimilarityResult)
                    and i < len(results["comparison_analyses"])
                    and isinstance(
                        results["comparison_analyses"][i], ResearchPaperAnalysis
                    )
                ):
                    ca = results["comparison_analyses"][i]
                    csv_data.append(
                        {
                            "Comparison_Paper_Title": ca.title,
                            "Overall_Similarity_Score": sr.final_similarity_score,
                            "Research_Question_Alignment": sr.research_question_alignment,
                            "Methodology_Alignment": sr.methodology_alignment,
                            "Findings_Alignment": sr.findings_alignment,
                            "Domain_Relevance": sr.domain_relevance,
                            "Conceptual_Overlap": sr.conceptual_overlap,
                            "Citation_Network_Overlap": sr.citation_network_overlap,
                            "Reasoning": sr.reasoning,
                        }
                    )
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                st.download_button(
                    label="Download Similarity Data (CSV)",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"semantic_similarity_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

    elif st.session_state.get("semantic_processing_status") == "processing":
        st.info("Analysis is currently in progress...")
    elif st.session_state.get("semantic_processing_status") == "error":
        st.error(
            "An error occurred during the last analysis attempt. Please check inputs or logs."
        )
    else:
        st.info(
            "Upload source and comparison PDF files and click 'Start Semantic Analysis'."
        )
