import os
import fitz # PyMuPDF
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_CONTENT_LENGTH = 50
DEFAULT_MIN_WORD_COUNT = 10
DEFAULT_MAX_WORKERS_PARAPHRASE = 4 # Specific to paraphrase
DEFAULT_BATCH_SIZE_PARAPHRASE = 5  # Specific to paraphrase

def is_meaningful_content(text: str, min_length: int, min_words: int) -> bool:
    """Check if text contains meaningful content worth processing."""
    if not text or len(text.strip()) < min_length:
        return False
    
    word_count = len(text.split())
    if word_count < min_words:
        return False
    
    # Check if it's mostly punctuation or numbers
    # Ensure len(text) is not zero to avoid DivisionByZero
    if len(text) == 0: return False
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / len(text) < 0.3:  # Less than 30% alphabetic characters
        return False
    
    non_content_patterns = [
        r'^\s*page\s+\d+\s*$', r'^\s*\d+\s*$', r'^\s*[^\w\s]+\s*$',
        r'^\s*table\s+of\s+contents\s*$',
    ]
    text_lower = text.lower().strip()
    for pattern in non_content_patterns:
        if re.match(pattern, text_lower):
            return False
    return True

def extract_text_from_file(file_content_bytes: bytes, file_type: str, 
                           min_content_length: int, min_word_count: int) -> str:
    """Extract text from PDF or TXT file content, with content validation."""
    # Note: Caching is handled by Streamlit's @st.cache_data in the UI layer if needed
    full_text = ""
    if file_type == "pdf":
        text_parts = []
        try:
            with fitz.open(stream=file_content_bytes, filetype="pdf") as doc:
                for page in doc:
                    page_text = page.get_text()
                    if not page_text.strip():
                        continue
                    
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = re.sub(r'-\s*\n\s*', '', page_text) # De-hyphenate
                    page_text = re.sub(r'\n+', '\n', page_text) # Normalize newlines
                    
                    if is_meaningful_content(page_text, min_content_length, min_word_count):
                        text_parts.append(page_text)
            full_text = '\n'.join(text_parts).strip()
        except Exception as e:
            # st.warning(f"Error extracting PDF text: {e}") # UI concern
            print(f"Warning: Error extracting PDF text: {e}") # Log to console
            return ""
            
    elif file_type == "txt":
        try:
            text = file_content_bytes.decode("utf-8", errors="ignore")
            text = re.sub(r'\s+', ' ', text) # Normalize whitespace
            full_text = text.strip()
        except Exception as e:
            # st.warning(f"Error processing TXT file: {e}") # UI concern
            print(f"Warning: Error processing TXT file: {e}")
            return ""
    
    if is_meaningful_content(full_text, min_content_length, min_word_count):
        return full_text
    return ""


def is_reference_section(section_title: str) -> bool:
    if not section_title: return False
    reference_keywords = [
        'reference', 'references', 'bibliography', 'works cited', 'citations',
        'literature cited', 'cited works', 'sources', 'further reading',
        'reading list', 'suggested reading', 'bibliographical', 'biblio',
        'acknowledgment', 'acknowledgments', 'appendix', 'appendices'
    ]
    title_lower = section_title.lower().strip()
    if any(keyword in title_lower for keyword in reference_keywords):
        return True
    if re.search(r'^\d+\.?\s*(reference|bibliography|works cited|appendix)', title_lower):
        return True
    return False

def chunk_text_by_sections(
    text_content: str, text_hash: str, 
    text_splitter: RecursiveCharacterTextSplitter,
    min_content_length: int, min_word_count: int
) -> List[Document]:
    if not text_content or not is_meaningful_content(text_content, min_content_length, min_word_count):
        return []
    
    section_patterns = [
        r"^\s*(\d+\.\d+\.?\s+[A-Z][^.]*)\s*$", r"^\s*(\d+\.?\s+[A-Z][^.]*)\s*$",
        r"^\s*([A-Z][A-Z\s]{2,})\s*$", r"^\s*(SECTION\s+\d+[^.]*)\s*$",
        r"^\s*(CHAPTER\s+\d+[^.]*)\s*$",
    ]
    lines = text_content.splitlines()
    sections_with_content = []
    temp_section_content = []
    current_section_title = "Document Content"
    skip_current_section = False
    section_found = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if not skip_current_section: temp_section_content.append(line)
            continue
        
        is_header = False
        header_text = None
        for pattern in section_patterns:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                header_text = match.group(1).strip()
                if 3 <= len(header_text) <= 100 and header_text.count(" ") <= 15:
                    is_header = True
                    section_found = True
                    break
        
        if is_header and header_text:
            if temp_section_content and not skip_current_section:
                content = "\n".join(temp_section_content).strip()
                if is_meaningful_content(content, min_content_length, min_word_count):
                    sections_with_content.append({"title": current_section_title, "content": content})
            temp_section_content = []
            current_section_title = header_text
            skip_current_section = is_reference_section(header_text)
            # if skip_current_section: st.info(f"Skipping reference section: {header_text}") # UI concern
        else:
            if not skip_current_section: temp_section_content.append(line)

    if temp_section_content and not skip_current_section:
        content = "\n".join(temp_section_content).strip()
        if is_meaningful_content(content, min_content_length, min_word_count):
            sections_with_content.append({"title": current_section_title, "content": content})

    if not section_found and not sections_with_content:
        if is_meaningful_content(text_content, min_content_length, min_word_count) and \
           not is_reference_section("Complete Document"):
            sections_with_content = [{"title": "Complete Document", "content": text_content.strip()}]
        else:
            # st.info("Document appears to contain no meaningful content for analysis") # UI concern
            return []

    final_chunks = []
    for section_info in sections_with_content:
        section_title = section_info["title"]
        section_content = section_info["content"]
        if not is_meaningful_content(section_content, min_content_length, min_word_count):
            continue
        try:
            chunks_from_section = text_splitter.split_text(section_content)
        except Exception: # Broad exception for robustness
            chunks_from_section = [section_content] # Fallback

        for i, chunk in enumerate(chunks_from_section):
            chunk_clean = chunk.strip()
            if is_meaningful_content(chunk_clean, min_content_length, min_word_count):
                final_chunks.append(Document(
                    page_content=chunk_clean,
                    metadata={
                        "section_title": section_title, "chunk_index_in_section": i + 1,
                        "source_doc_hash": text_hash, "word_count": len(chunk_clean.split()),
                        "char_count": len(chunk_clean)
                    }))
    return final_chunks

def create_vector_store_for_paraphrase(docs: List[Document], embeddings_model, 
                                       min_content_length: int, min_word_count: int) -> Optional[FAISS]:
    if not docs: return None
    valid_docs = [doc for doc in docs if is_meaningful_content(doc.page_content, min_content_length, min_word_count)]
    if not valid_docs: return None
    try:
        return FAISS.from_documents(valid_docs, embeddings_model)
    except Exception as e:
        # st.warning(f"Failed to create vector store: {e}") # UI concern
        print(f"Warning: Failed to create vector store: {e}")
        return None

def calculate_text_similarity_jaccard(text1: str, text2: str) -> float:
    if not text1 or not text2: return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2: return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def _process_single_chunk_for_paraphrase(
    source_doc_chunk: Document, 
    comparison_stores: Dict[str, FAISS], 
    chat_client, # Pass the initialized client
    min_content_length: int, 
    min_word_count: int
) -> Optional[Dict]:
    source_content = source_doc_chunk.page_content.strip()
    source_metadata = source_doc_chunk.metadata
    
    if not is_meaningful_content(source_content, min_content_length, min_word_count):
        return None

    section_title = source_metadata.get("section_title", "Unknown Section")
    chunk_in_section_idx = source_metadata.get("chunk_index_in_section", 0)

    for comp_filename, comp_vector_store in comparison_stores.items():
        if comp_vector_store is None: continue
        try:
            results = comp_vector_store.similarity_search_with_score(source_content, k=5)
            best_match_text = None
            best_score = float('inf')
            best_word_sim = 0.0
            
            for doc, score in results:
                comparison_text = doc.page_content.strip()
                if not is_meaningful_content(comparison_text, min_content_length, min_word_count):
                    continue
                
                word_similarity = calculate_text_similarity_jaccard(source_content, comparison_text)
                
                if score < 0.4 or word_similarity > 0.25: # Thresholds from original
                    if score < best_score:
                        best_match_text = comparison_text
                        best_score = score
                        best_word_sim = word_similarity

            if best_match_text:
                prompt = f"""Compare these texts briefly:

SOURCE: {source_content[:500]}...
COMPARISON: {best_match_text[:500]}...

If similar/paraphrased, respond: MATCH: YES | REASON: [brief reason]
If not similar, respond: MATCH: NO"""
                try:
                    response = chat_client.invoke(prompt)
                    content = response.content.strip()

                    if "MATCH: YES" in content.upper():
                        reason_match = re.search(r'REASON:\s*([^\n]+)', content, re.IGNORECASE)
                        reason = reason_match.group(1).strip() if reason_match else "Similar content detected"
                        return {
                            "source_section_title": section_title,
                            "source_chunk_index": chunk_in_section_idx,
                            "source_text": source_content,
                            "matched_file": comp_filename,
                            "matched_text": best_match_text[:300] + "..." if len(best_match_text) > 300 else best_match_text,
                            "reason": reason,
                            "vector_score": best_score,
                            "word_similarity": best_word_sim
                        }
                except Exception as e:
                    print(f"LLM call failed for chunk comparison: {e}") # Log error
                    continue # Try next comparison store or chunk
        except Exception as e:
            print(f"Similarity search failed for {comp_filename}: {e}") # Log error
            continue
    return None


def detect_paraphrased_sections_processing(
    source_documents: List[Document], 
    comparison_vector_stores_map: Dict[str, FAISS],
    chat_client, # Pass initialized client
    min_content_length: int, 
    min_word_count: int,
    batch_size: int,
    max_workers: int,
    progress_callback = None # For Streamlit progress updates
) -> List[Dict]:
    if not source_documents or not comparison_vector_stores_map:
        return []
    
    meaningful_docs = [doc for doc in source_documents if is_meaningful_content(doc.page_content, min_content_length, min_word_count)]
    if not meaningful_docs:
        return []

    detected_paraphrases = []
    total_docs = len(meaningful_docs)
    processed_docs = 0

    with ThreadPoolExecutor(max_workers=min(max_workers, os.cpu_count() or 1)) as executor:
        futures = {
            executor.submit(
                _process_single_chunk_for_paraphrase, 
                doc, 
                comparison_vector_stores_map, 
                chat_client,
                min_content_length,
                min_word_count
            ): doc for doc in meaningful_docs
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    detected_paraphrases.append(result)
            except Exception as e:
                # doc_info = futures[future].metadata.get("section_title", "Unknown chunk") # Example
                print(f"Error processing a source chunk: {e}") # Log error
            finally:
                processed_docs += 1
                if progress_callback:
                    progress_callback(processed_docs / total_docs)
                    
    return detected_paraphrases


def load_comparison_docs_for_paraphrase(
    directory_path: str, 
    text_splitter: RecursiveCharacterTextSplitter, 
    embeddings_model, # Pass initialized model
    file_hash_func, # Pass get_file_hash from core_utils
    min_content_length: int, 
    min_word_count: int,
    progress_callback = None # For Streamlit progress
) -> Dict[str, FAISS]:
    comparison_stores = {}
    if not os.path.isdir(directory_path):
        return comparison_stores
    
    files_to_process = [f for f in os.listdir(directory_path) if f.lower().endswith(('.pdf', '.txt'))]
    if not files_to_process:
        return comparison_stores
    
    total_files = len(files_to_process)
    for i, filename in enumerate(files_to_process):
        if progress_callback:
            progress_callback(filename, (i + 1) / total_files)
        
        try:
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "rb") as f:
                file_content = f.read()

            file_hash = file_hash_func(file_content) # Use passed hash function
            file_extension = filename.split(".")[-1].lower()
            
            extracted_text = extract_text_from_file(file_content, file_extension, min_content_length, min_word_count)

            if not extracted_text:  # Already checks for meaningful content
                continue

            chunks = text_splitter.split_text(extracted_text)
            docs = [Document(page_content=chunk.strip()) 
                   for chunk in chunks 
                   if is_meaningful_content(chunk.strip(), min_content_length, min_word_count)]

            if docs:
                vector_store = create_vector_store_for_paraphrase(docs, embeddings_model, min_content_length, min_word_count)
                if vector_store:
                    comparison_stores[filename] = vector_store
                   

        except Exception as e:
            # st.warning(f"Error processing {filename}: {e}") # UI concern
            print(f"Warning: Error processing comparison file {filename}: {e}")
            continue
    
    return comparison_stores