import streamlit as st
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core_utils import get_file_hash, PARAPHRASE_CHAT_MODEL, PARAPHRASE_EMBEDDING_MODEL
from .paraphrase_processing import (
    extract_text_from_file, chunk_text_by_sections,
    load_comparison_docs_for_paraphrase, detect_paraphrased_sections_processing,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MIN_CONTENT_LENGTH,
    DEFAULT_MIN_WORD_COUNT, DEFAULT_MAX_WORKERS_PARAPHRASE, DEFAULT_BATCH_SIZE_PARAPHRASE
)

# Cache the text splitter instance
@st.cache_resource
def get_recursive_text_splitter_paraphrase(chunk_size: int, chunk_overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def render_paraphrase_detector_ui(chat_client, embeddings_model):
    st.header("Contextual Paraphrase Detector")
    st.markdown("Upload a source document and specify a directory of comparison documents to detect potential paraphrasing.")
    st.markdown("---")

    # Configuration is local to this UI function's invocation
    # These will be passed to processing functions
    st.sidebar.subheader("Paraphrase Detection Settings")
    chunk_size_ui = st.sidebar.slider("Chunk Size", 200, 2000, DEFAULT_CHUNK_SIZE, 100, key="para_chunk_size")
    chunk_overlap_ui = st.sidebar.slider("Chunk Overlap", 50, 500, DEFAULT_CHUNK_OVERLAP, 50, key="para_chunk_overlap")
    min_content_length_ui = st.sidebar.slider("Min Content Length (chars)", 30, 200, DEFAULT_MIN_CONTENT_LENGTH, 10, key="para_min_len")
    min_word_count_ui = st.sidebar.slider("Min Word Count", 5, 50, DEFAULT_MIN_WORD_COUNT, 5, key="para_min_words")
    batch_size_ui = st.sidebar.slider("Processing Batch Size (source chunks)", 1, 20, DEFAULT_BATCH_SIZE_PARAPHRASE, 1, key="para_batch_size") # Note: current processing is per chunk
    max_workers_ui = st.sidebar.slider("Max Parallel Workers", 1, 10, DEFAULT_MAX_WORKERS_PARAPHRASE, 1, key="para_max_workers")


    source_file = st.file_uploader(
        "Upload Your Source Document (PDF or TXT)",
        type=["pdf", "txt"],
        key="paraphrase_source_uploader",
    )

    # Default path for example, user should change this
    default_comparison_dir = os.path.join(os.path.expanduser("~"), "Desktop", "comparison_docs_paraphrase")
    if not os.path.exists(default_comparison_dir):
        try:
            os.makedirs(default_comparison_dir)
            st.info(f"Created a default comparison directory at: {default_comparison_dir}. Please add your comparison PDF/TXT files there.")
        except Exception as e:
            st.warning(f"Could not create default directory {default_comparison_dir}: {e}. Please specify a valid path.")
            default_comparison_dir = ""


    comparison_docs_directory = st.text_input(
        "Path to Directory of Documents to Compare Against",
        value=default_comparison_dir, # Provide a sensible default or leave empty
        key="paraphrase_doc_dir_input",
        help="Enter the full path to a folder containing PDF/TXT files."
    )

    st.markdown("---")

    if st.button("Start Paraphrase Detection", type="primary", key="paraphrase_start_button"):
        if not source_file:
            st.warning("Please upload a source document.")
            st.stop()
        
        if not comparison_docs_directory or not os.path.isdir(comparison_docs_directory):
            st.warning("Please provide a valid directory path for comparison documents.")
            st.stop()

        # Get the configured text splitter
        text_splitter = get_recursive_text_splitter_paraphrase(chunk_size_ui, chunk_overlap_ui)

        # Process Source Document
        with st.spinner("Processing source document..."):
            source_file_bytes = source_file.getvalue()
            source_file_hash = get_file_hash(source_file_bytes)
            source_file_type = source_file.type.split("/")[-1].lower() if source_file.type else source_file.name.split(".")[-1].lower()
            
            # Use @st.cache_data for functions that return data if they are pure and inputs are hashable
            # For this integration, direct call:
            source_text = extract_text_from_file(source_file_bytes, source_file_type, min_content_length_ui, min_word_count_ui)
            
            if not source_text:
                st.error("Failed to extract meaningful text from source document or content is too short/irrelevant.")
                st.stop()

            source_documents = chunk_text_by_sections(source_text, source_file_hash, text_splitter, min_content_length_ui, min_word_count_ui)

            if not source_documents:
                st.warning("No meaningful content available in source document for analysis after filtering and chunking.")
                st.stop()
        st.success(f"Source document processed: {len(source_documents)} chunks ready for analysis.")

        # Load Comparison Documents
        comparison_progress_bar = st.progress(0)
        comparison_status_text = st.empty()
        def comp_load_progress_callback(filename, progress):
            comparison_status_text.text(f"Loading comparison document: {filename} ({progress*100:.0f}%)")
            comparison_progress_bar.progress(progress)

        with st.spinner("Loading and vectorizing comparison documents..."):
            comparison_stores = load_comparison_docs_for_paraphrase(
                comparison_docs_directory, text_splitter, embeddings_model, get_file_hash,
                min_content_length_ui, min_word_count_ui, comp_load_progress_callback
            )
        comparison_progress_bar.empty()
        comparison_status_text.empty()

        if not comparison_stores:
            st.error(f"No valid comparison documents found or processed in '{comparison_docs_directory}'.")
            st.stop()
        st.success(f"Loaded and vectorized {len(comparison_stores)} comparison documents.")

        # Detect Paraphrases
        detection_progress_bar = st.progress(0)
        detection_status_text = st.empty()
        def detection_progress_callback(progress):
            detection_status_text.text(f"Detecting paraphrases... ({progress*100:.0f}%)")
            detection_progress_bar.progress(progress)

        with st.spinner("Detecting paraphrases (this may take a while)..."):
            detected_paraphrases = detect_paraphrased_sections_processing(
                source_documents, comparison_stores, chat_client,
                min_content_length_ui, min_word_count_ui,
                batch_size_ui, max_workers_ui, # batch_size_ui is for conceptual batching, actual is per chunk
                detection_progress_callback
            )
        detection_progress_bar.empty()
        detection_status_text.empty()

        # Display Results
        st.markdown("---")
        st.subheader("Paraphrase Detection Results")
        if detected_paraphrases:
            st.success(f"Found {len(detected_paraphrases)} potential paraphrases.")
            
            # Summary statistics
            # Ensure all required keys exist before calculating mean
            valid_vector_scores = [m['vector_score'] for m in detected_paraphrases if 'vector_score' in m and m['vector_score'] is not None]
            valid_word_sim = [m['word_similarity'] for m in detected_paraphrases if 'word_similarity' in m and m['word_similarity'] is not None]

            avg_vector_score = np.mean(valid_vector_scores) if valid_vector_scores else 0
            avg_word_sim = np.mean(valid_word_sim) if valid_word_sim else 0
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Matches", len(detected_paraphrases))
            with col2: st.metric("Avg Vector Score", f"{avg_vector_score:.3f}")
            with col3: st.metric("Avg Word Similarity", f"{avg_word_sim:.3f}")
            st.markdown("---")

            for i, match in enumerate(detected_paraphrases):
                expander_title = f"Match {i+1}: In '{match.get('source_section_title', 'N/A')}' (Chunk {match.get('source_chunk_index', 'N/A')}) vs '{match.get('matched_file', 'N/A')}'"
                with st.expander(expander_title):
                    col_src, col_match_txt = st.columns([1, 1])
                    with col_src:
                        st.markdown("**Source Content:**")
                        st.text_area("Source", value=match.get("source_text", ""), height=200, disabled=True, key=f"para_source_{i}")
                    with col_match_txt:
                        st.markdown(f"**Matched Content from: {match.get('matched_file', 'N/A')}**")
                        st.text_area("Matched", value=match.get("matched_text", ""), height=200, disabled=True, key=f"para_match_{i}")
                    
                    st.markdown("**Analysis:**")
                    # Use two columns for reason and scores for better layout
                    col_reason, col_scores = st.columns(2)
                    with col_reason:
                        st.info(f"**Reason:** {match.get('reason', 'N/A')}")
                    with col_scores:
                        st.info(f"**Scores:** Vector: {match.get('vector_score', 0.0):.3f}, Word Sim: {match.get('word_similarity', 0.0):.3f}")
        else:
            st.info("No paraphrased content detected based on the current settings and documents.")
            st.markdown("""
            This could mean:
            - The documents are original.
            - The similarity thresholds or content filters are too strict.
            - The content is too different for the current detection method.
            - An issue occurred during processing (check console logs if running locally).
            """)
    else:
        st.info("Upload a source document, specify a comparison directory, and click 'Start Paraphrase Detection'.")

    st.sidebar.markdown("---")
    with st.sidebar.expander("Paraphrase Detection Performance Tips"):
        st.markdown("""
        - Use smaller chunk sizes for more granular analysis, but this increases processing time.
        - Increase minimum content length/word count to filter out noise effectively.
        - Ensure comparison documents are relevant and contain substantial text.
        - For very large documents or many comparison files, processing can be lengthy.
        """)