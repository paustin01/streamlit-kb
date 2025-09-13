"""
GenAI Bedrock Knowledgebase Application
=====================================

This Streamlit application creates a local knowledgebase system that allows users to:
1. Upload documents (PDF, TXT, MD files)
2. Convert them into searchable vector embeddings using AWS Bedrock
3. Ask questions about the documents using natural language
4. Get AI-powered answers with source citations

Key Components:
- Document Processing: Extracts text from PDFs and processes text files
- Vector Storage: Uses ChromaDB to store document embeddings locally
- AI Integration: Uses AWS Bedrock Nova Micro LLM for question answering
- Web Interface: Streamlit provides an easy-to-use web interface

Prerequisites:
- AWS credentials configured with Bedrock access
- Python packages: streamlit, langchain, PyPDF2, chromadb, boto3
- .env file with AWS configuration (optional)

How to Use:
1. Run: streamlit run app.py
2. Upload documents via the "Upload Files" tab
3. Click "Re-index Knowledgebase" to process documents
4. Ask questions in the "Ask Questions" tab
5. Manage files in the "Delete Files" tab
"""

# --- IMPORTS ---
# Core Python libraries
import os
import shutil
import stat
import time
import tempfile
import atexit
import sqlite3
import uuid
import hashlib
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List
import math

# Streamlit for web interface
import streamlit as st

# Environment and PDF processing
from dotenv import load_dotenv
import PyPDF2

# LangChain components for AI and document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA

# --- ENVIRONMENT SETUP ---
# Load environment variables from .env file (AWS credentials, etc.)
load_dotenv()

# --- CONFIGURATION CONSTANTS ---
# Directory where uploaded documents are stored
DATA_DIR = "data"

# SQLite database for chat history
CHAT_DB_PATH = "chat_history.db"

# Persistent ChromaDB directory to survive hot reloads
CHROMA_DIR = "chroma_db"

# Create a temporary directory for ChromaDB that gets cleaned up automatically
# This avoids permission issues and keeps the workspace clean
# TEMP_DIR = tempfile.mkdtemp(prefix="knowledgebase_chromadb_")
# CHROMA_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Register cleanup function to remove temp directory when app exits
# def cleanup_temp_dir():
#     """Clean up temporary directory when application exits"""
#     try:
#         if os.path.exists(TEMP_DIR):
#             shutil.rmtree(TEMP_DIR, ignore_errors=True)
#             print(f"✅ Cleaned up temporary directory: {TEMP_DIR}")
#     except Exception as e:
#         print(f"⚠️ Could not clean up temporary directory: {e}")

# atexit.register(cleanup_temp_dir)

# AWS Bedrock model IDs - these are the AI models we'll use
AWS_BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # For converting text to vectors
AWS_BEDROCK_LLM_MODEL_ID = "us.amazon.nova-premier-v1:0"          # For answering questions
AWS_REGION = "us-west-2"  # AWS region where Bedrock is available

# --- DATABASE FUNCTIONS ---
def init_chat_database():
    """
    Initialize the SQLite database for storing chat history
    Creates tables if they don't exist
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()

    # Create chat_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create chat_messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_type TEXT,  -- 'user' or 'assistant'
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    ''')

    conn.commit()
    conn.close()

def create_chat_session(title=None):
    """
    Create a new chat session

    Args:
        title (str): Optional title for the session

    Returns:
        str: Session ID
    """
    session_id = str(uuid.uuid4())
    if not title:
        title = f"New Chat"

    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (session_id, title)
        VALUES (?, ?)
    ''', (session_id, title))
    conn.commit()
    conn.close()

    return session_id

def update_chat_title(session_id, new_title):
    """
    Update the title of a chat session

    Args:
        session_id (str): Session ID
        new_title (str): New title for the session
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE chat_sessions 
        SET title = ?, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
    ''', (new_title, session_id))
    conn.commit()
    conn.close()

def save_chat_message(session_id, message_type, content):
    """
    Save a chat message to the database

    Args:
        session_id (str): Session ID
        message_type (str): 'user' or 'assistant'
        content (str): Message content
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_messages (session_id, message_type, content)
        VALUES (?, ?, ?)
    ''', (session_id, message_type, content))

    # Update session timestamp
    cursor.execute('''
        UPDATE chat_sessions
        SET updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
    ''', (session_id,))

    conn.commit()
    conn.close()

def get_chat_sessions():
    """
    Get all chat sessions ordered by most recent

    Returns:
        list: List of session tuples (session_id, title, updated_at)
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT session_id, title, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
    ''')
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_chat_messages(session_id):
    """
    Get all messages for a specific chat session

    Args:
        session_id (str): Session ID

    Returns:
        list: List of message tuples (message_type, content, timestamp)
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT message_type, content, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

def delete_chat_session(session_id):
    """
    Delete a chat session and all its messages

    Args:
        session_id (str): Session ID to delete
    """
    conn = sqlite3.connect(CHAT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
    cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

# --- VECTORSTORE PERSISTENCE FUNCTIONS ---
def get_data_hash():
    """
    Generate a hash of all data files to detect changes

    Returns:
        str: MD5 hash of all file contents and timestamps
    """
    hash_md5 = hashlib.md5()

    if not os.path.exists(DATA_DIR):
        return hash_md5.hexdigest()

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    files.sort()  # Ensure consistent ordering

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            # Hash filename and modification time
            hash_md5.update(filename.encode('utf-8'))
            hash_md5.update(str(os.path.getmtime(file_path)).encode('utf-8'))

            # Hash file contents
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception:
            continue

    return hash_md5.hexdigest()

def save_vectorstore_metadata(data_hash, chroma_dir):
    """
    Save metadata about the vectorstore

    Args:
        data_hash (str): Hash of the data used to create vectorstore
        chroma_dir (str): Path to the ChromaDB directory
    """
    metadata = {
        'data_hash': data_hash,
        'chroma_dir': chroma_dir,
        'created_at': datetime.now().isoformat()
    }

    with open('vectorstore_metadata.json', 'w') as f:
        json.dump(metadata, f)

def load_vectorstore_metadata():
    """
    Load vectorstore metadata

    Returns:
        dict: Metadata dictionary or None if not found
    """
    try:
        with open('vectorstore_metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None

def vectorstore_needs_update():
    """
    Check if vectorstore needs to be updated based on data changes

    Returns:
        bool: True if update needed, False otherwise
    """
    current_hash = get_data_hash()
    metadata = load_vectorstore_metadata()

    if not metadata:
        return True

    # Check if ChromaDB directory still exists
    if not os.path.exists(metadata.get('chroma_dir', '')):
        return True

    return current_hash != metadata.get('data_hash')

def load_existing_vectorstore():
    """
    Load existing vectorstore if it exists and is valid

    Returns:
        bool: True if successfully loaded, False otherwise
    """
    metadata = load_vectorstore_metadata()

    if not metadata:
        return False

    chroma_dir = metadata.get('chroma_dir')
    if not chroma_dir or not os.path.exists(chroma_dir):
        return False

    try:
        st.session_state.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embedding_model
        )

        # Test the vectorstore
        test_results = st.session_state.vectorstore.similarity_search("test", k=1)
        return True

    except Exception as e:
        print(f"Error loading existing vectorstore: {e}")
        return False

# --- AI COMPONENTS INITIALIZATION ---
# Initialize the embedding model (converts text to numerical vectors for similarity search)
embedding_model = BedrockEmbeddings(
    region_name="us-west-2",
    model_id=AWS_BEDROCK_EMBEDDING_MODEL_ID
)

# Initialize text splitter (breaks documents into smaller chunks for processing)
# chunk_size=500: Each chunk is ~500 characters
# chunk_overlap=50: Overlapping chunks to maintain context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- PDF PROCESSING FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using PyPDF2
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content, or None if extraction fails
    """
    text = ""
    
    try:
        # Open PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Loop through each page in the PDF
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Add double newline between pages
                    
    except Exception as e:
        st.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None
    
    return text.strip()

def convert_pdf_to_text(uploaded_file, filename):
    """
    Convert an uploaded PDF file to text and save it as a .txt file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        filename (str): Original filename of the uploaded PDF
        
    Returns:
        tuple: (text_filename, text_length) or (None, 0) if conversion fails
    """
    try:
        # Step 1: Save the uploaded PDF temporarily
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Step 2: Extract text from the temporary PDF
        text = extract_text_from_pdf(temp_pdf_path)
        
        # Step 3: Clean up the temporary PDF file
        os.remove(temp_pdf_path)
        
        if text:
            # Step 4: Save extracted text as a .txt file
            text_filename = filename.replace('.pdf', '.txt')
            text_path = os.path.join(DATA_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            return text_filename, len(text)
        else:
            return None, 0
            
    except Exception as e:
        st.error(f"Error processing PDF {filename}: {e}")
        # Clean up temporary file if something went wrong
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return None, 0

# --- DATABASE MANAGEMENT FUNCTIONS ---
def process_file_chunk(args):
    """
    Process a single file for the vectorstore (used for concurrent processing)

    Args:
        args: Tuple of (filename, file_path, text_splitter)

    Returns:
        tuple: (filename, document_chunks) or (filename, None) if error
    """
    filename, file_path, text_splitter = args

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if text.strip():  # Only process non-empty files
            # Split text into chunks using the text splitter with metadata
            splits = text_splitter.create_documents(
                [text], 
                metadatas=[{"source": filename}]
            )
            return (filename, splits)
        else:
            return (filename, None)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return (filename, None)

def create_embeddings_batch(docs_batch: List, embedding_model, batch_id: int, total_batches: int):
    """
    Create embeddings for a batch of documents

    Args:
        docs_batch: List of documents to embed
        embedding_model: The embedding model to use
        batch_id: Current batch number
        total_batches: Total number of batches

    Returns:
        tuple: (batch_id, embeddings_list, texts_list) or (batch_id, None, None) on error
    """
    try:
        texts = [doc.page_content for doc in docs_batch]

        # Create embeddings for the batch
        embeddings = embedding_model.embed_documents(texts)

        return (batch_id, embeddings, texts)

    except Exception as e:
        print(f"Error creating embeddings for batch {batch_id}: {e}")
        return (batch_id, None, None)

def create_vectorstore_with_progress(docs: List, embedding_model, persist_directory: str,
                                     batch_size: int = 50, progress_callback=None):
    """
    Create vectorstore with real progress tracking by processing documents in batches

    Args:
        docs: List of documents to embed
        embedding_model: The embedding model
        persist_directory: Where to save the vectorstore
        batch_size: Number of documents per batch
        progress_callback: Function to call for progress updates

    Returns:
        Chroma vectorstore instance
    """
    import time
    import threading

    total_docs = len(docs)
    print(f"Processing {total_docs} documents in batches of {batch_size}")

    if progress_callback:
        progress_callback(0.0, f"Initializing vectorstore...")

    # Create empty vectorstore first
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # Process documents in batches to show progress
    processed_docs = 0

    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = docs[i:batch_end]
        batch_size_actual = len(batch_docs)

        if progress_callback:
            progress = processed_docs / total_docs
            progress_callback(progress, f"Processing batch {i//batch_size + 1}: docs {i+1}-{batch_end}")

        # Add batch to vectorstore
        try:
            vectorstore.add_documents(batch_docs)
            processed_docs += batch_size_actual

            # Small delay to show progress updates
            time.sleep(0.1)

        except Exception as e:
            print(f"Error adding batch {i//batch_size + 1}: {e}")
            # Continue with next batch
            continue

    if progress_callback:
        progress_callback(1.0, f"Completed! Processed {processed_docs} documents")

    print(f"Successfully created vectorstore with {processed_docs} documents")
    return vectorstore

def safe_remove_directory(directory):
    """
    Safely remove a directory with proper error handling and permission fixes

    Args:
        directory (str): Path to directory to remove

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(directory):
        return True

    try:
        # Fix permissions recursively before deletion
        for root, dirs, files in os.walk(directory):
            # Make directories writable
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.chmod(dir_path, 0o755)
                except Exception:
                    pass

            # Make files writable
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    os.chmod(file_path, 0o644)
                except Exception:
                    pass

        # Make root directory writable
        os.chmod(directory, 0o755)

        # Now remove the directory
        shutil.rmtree(directory, ignore_errors=True)
        return not os.path.exists(directory)

    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
        # Try force removal as last resort
        try:
            shutil.rmtree(directory, ignore_errors=True)
            return not os.path.exists(directory)
        except:
            return False

# --- CORE KNOWLEDGEBASE FUNCTION ---
def reindex_knowledgebase():
    """
    Re-index the knowledgebase by processing all documents in the data directory
    This function now includes:
    1. Progress tracking with real-time updates
    2. Concurrent file processing for better performance
    3. Persistent storage that survives hot reloads
    4. Smart checking to avoid unnecessary re-indexing

    Returns:
        bool: True if successful, False otherwise
    """

    # Step 1: Check if we actually need to reindex
    if not vectorstore_needs_update():
        if load_existing_vectorstore():
            st.info("📚 Vectorstore is up to date, no reindexing needed!")
            st.session_state.vectorstore_loaded = True
            return True

    # Step 2: Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Step 3: Get list of files to process
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]

    if not files:
        st.error("No valid documents found to index.")
        return False

    # Step 4: Setup progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(files) + 2  # +2 for vectorstore creation and testing
    current_step = 0

    try:
        status_text.text("🔍 Processing files...")

        # Step 5: Process files concurrently with progress tracking
        docs = []
        processed_files = []

        # Prepare arguments for concurrent processing
        file_args = [
            (filename, os.path.join(DATA_DIR, filename), text_splitter)
            for filename in files
        ]

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
            # Submit all file processing tasks
            future_to_filename = {
                executor.submit(process_file_chunk, args): args[0]
                for args in file_args
            }
            # Process completed tasks as they finish
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress)

                try:
                    result_filename, splits = future.result()
                    if splits:
                        docs.extend(splits)
                        processed_files.append(result_filename)
                        status_text.text(f"✅ Processed {result_filename} ({len(splits)} chunks)")
                    else:
                        status_text.text(f"⚠️ Skipped empty file: {result_filename}")

                except Exception as e:
                    st.warning(f"Error processing {filename}: {e}")

                time.sleep(0.1)  # Small delay to show progress

        # Step 6: Check if we have any documents to process
        if not docs:
            st.error("No valid documents found to index after processing.")
            return False

        # Step 7: Create vectorstore with batched embeddings and progress tracking
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        status_text.text(f"🔮 Creating embeddings for {len(docs)} document chunks...")

        # Remove old vectorstore if it exists
        if os.path.exists(CHROMA_DIR):
            safe_remove_directory(CHROMA_DIR)

        # Create directory with proper permissions
        os.makedirs(CHROMA_DIR, exist_ok=True)
        os.chmod(CHROMA_DIR, 0o755)

        # Progress callback for embedding creation
        embedding_progress_bar = st.progress(0)
        embedding_status = st.empty()

        def embedding_progress_callback(progress: float, message: str):
            embedding_progress_bar.progress(progress)
            embedding_status.text(f"⚡ {message}")

        # Create vectorstore with real progress tracking
        st.session_state.vectorstore = create_vectorstore_with_progress(
            docs=docs,
            embedding_model=embedding_model,
            persist_directory=CHROMA_DIR,
            batch_size=50,  # Process 50 documents at a time
            progress_callback=embedding_progress_callback
        )

        # Clean up embedding progress indicators
        embedding_progress_bar.empty()
        embedding_status.empty()

        # Step 8: Test the vectorstore
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        status_text.text("🔬 Testing vectorstore...")

        try:
            test_results = st.session_state.vectorstore.similarity_search("test", k=1)
            status_text.text("✅ Vectorstore test successful!")
        except Exception as e:
            st.error(f"Vectorstore test failed: {e}")
            return False

        # Step 9: Save metadata for persistence
        data_hash = get_data_hash()
        save_vectorstore_metadata(data_hash, CHROMA_DIR)

        # Step 10: Show final success messages
        progress_bar.progress(1.0)
        st.success(f"✅ Successfully processed {len(processed_files)} files: {', '.join(processed_files)}")
        st.success(f"✅ Created {len(docs)} document chunks")
        st.success(f"✅ Vectorstore saved to {CHROMA_DIR}")

        # Clean up UI elements
        progress_bar.empty()
        status_text.empty()

        # Celebrate!
        time.sleep(1)
        st.balloons()

        return True

    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        st.error("Please check your AWS credentials and Bedrock access.")

        # Clean up UI elements on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

        return False

# --- STREAMLIT APPLICATION SETUP ---
# Initialize database
init_chat_database()

# Configure the main page
st.set_page_config(page_title="☢️ Gigawatt chatbot")

# Add header image
try:
    st.image("header.jpeg", use_container_width=True)
except:
    # Fallback if image not found
    st.title("☢️ Gigawatt chatbot")

# --- NAVIGATION SIDEBAR ---
# Initialize session state for page navigation (remembers which page user is on)
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Chat"

# Initialize vectorstore loading state
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

# Auto-load existing vectorstore on startup if available
if not st.session_state.vectorstore_loaded and 'vectorstore' not in st.session_state:
    if load_existing_vectorstore():
        st.session_state.vectorstore_loaded = True

# Initialize chat session state
if 'current_chat_session' not in st.session_state:
    st.session_state.current_chat_session = None

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

st.sidebar.title("🧭 Navigation")

# Create navigation buttons for different pages
if st.sidebar.button("💬 Chat", use_container_width=True):
    st.session_state.current_page = "Chat"

if st.sidebar.button("📤 Upload Files / Re-Index", use_container_width=True):
    st.session_state.current_page = "Upload Files"

if st.sidebar.button("🗑️ Delete Files", use_container_width=True):
    st.session_state.current_page = "Delete Files"

# --- CHAT SESSION MANAGEMENT ---
st.sidebar.markdown("---")
st.sidebar.title("💬 Chat Sessions")

# New chat button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_session_id = create_chat_session()
    st.session_state.current_chat_session = new_session_id
    st.session_state.chat_messages = []
    st.session_state.current_page = "Chat"
    st.rerun()

# Load existing sessions
chat_sessions = get_chat_sessions()
if chat_sessions:
    st.sidebar.subheader("💬 Recent Chats")
    for session_id, title, updated_at in chat_sessions[:10]:  # Show last 10 sessions
        # Create columns for chat and delete button
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            # Truncate title if too long
            display_title = title if len(title) <= 30 else f"{title[:27]}..."
            button_key = f"load_session_{session_id}"
            if st.button(f"💭 {display_title}", key=button_key, use_container_width=True):
                st.session_state.current_chat_session = session_id
                st.session_state.chat_messages = get_chat_messages(session_id)
                st.session_state.current_page = "Chat"
                st.rerun()
        
        with col2:
            # Delete session option
            delete_key = f"delete_session_{session_id}"
            if st.button("🗑️", key=delete_key, help="Delete this chat"):
                delete_chat_session(session_id)
                if st.session_state.current_chat_session == session_id:
                    st.session_state.current_chat_session = None
                    st.session_state.chat_messages = []
                st.rerun()

# Get the current page from session state
page = st.session_state.current_page

# Load chat messages for current session if we have one
if st.session_state.current_chat_session and not st.session_state.chat_messages:
    st.session_state.chat_messages = get_chat_messages(st.session_state.current_chat_session)

# --- PAGE 1: UPLOAD FILES ---
if page == "Upload Files":
    st.header("📤 Upload Files")
    
    # Instructions for users
    st.info("💡 **Note**: After uploading files, click 'Re-index Knowledgebase' to make them searchable.")

    # Re-index button (this is the most important button!)
    if st.button("🔄 Re-index Knowledgebase"):
        with st.spinner("Re-indexing..."):
            success = reindex_knowledgebase()
        if success:
            st.success("Knowledgebase re-indexed successfully!")
            st.session_state.vectorstore_loaded = True
            # Force reload of vectorstore in session state
            st.session_state.vectorstore_initialized = True
            print(f"📁 Vectorstore initialized in session state: {st.session_state.vectorstore_initialized}")
            st.rerun()  # Refresh the page to show updated status
        else:
            st.warning("No documents found to index or indexing failed.")
            st.session_state.vectorstore_loaded = False

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # File upload widget
    accepted_types = [".txt", ".md", ".pdf"]
    uploaded_files = st.file_uploader(
        f"Upload {', '.join(accepted_types).upper()} files",
        type=accepted_types,
        accept_multiple_files=True
    )

    # Show existing files in the data directory
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if existing_files:
        st.subheader("📋 Current Files")
        for file in existing_files:
            st.write(f"• {file}")
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.pdf'):
                    # Handle PDF files - convert to text
                    with st.spinner(f"Processing PDF: {file.name}..."):
                        text_filename, text_length = convert_pdf_to_text(file, file.name)
                        if text_filename:
                            st.success(f"✅ Converted {file.name} to {text_filename} ({text_length} characters)")
                        else:
                            st.error(f"❌ Failed to extract text from {file.name}")
                else:
                    # Handle text/markdown files - save directly
                    save_path = os.path.join(DATA_DIR, file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.read())
                    st.success(f"✅ Saved {file.name} to /data")
                    
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

# --- PAGE 2: DELETE FILES ---
elif page == "Delete Files":
    st.header("🗑️ Delete Files")
    
    # Show current knowledgebase status
    if st.session_state.vectorstore_loaded:
        st.info("✅ Knowledgebase is loaded and ready to use")
    else:
        st.warning("⚠️ No knowledgebase found.")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Get list of files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    
    if not files:
        st.info("📁 No files found in the data directory.")
    else:
        st.subheader("📋 Available Files")
        
        # Show files with checkboxes for selection
        selected_files = st.multiselect(
            "Select files to delete:",
            options=files,
            help="Choose one or more files to delete from your knowledgebase"
        )
        
        # Show selected files for confirmation
        if selected_files:
            st.write("**Files selected for deletion:**")
            for file in selected_files:
                file_type = "📄 PDF→TXT" if file.endswith('.txt') and file.replace('.txt', '.pdf') else "📝 TEXT"
                st.write(f"• ❌ {file} ({file_type})")
            
            # Confirmation button
            if st.button("🗑️ Delete Selected Files", type="primary"):
                deleted_count = 0
                for f in selected_files:
                    try:
                        file_path = os.path.join(DATA_DIR, f)
                        os.remove(file_path)
                        st.success(f"✅ Deleted: {f}")
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"❌ Error deleting {f}: {e}")
                
                if deleted_count > 0:
                    st.success(f"🎉 Successfully deleted {deleted_count} file(s)")
                    st.balloons()
                    
                    # Auto-refresh the page to update file list
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("👆 Select files above to delete them")
        
        # Show all current files for reference
        st.markdown("---")
        st.subheader("📂 All Current Files")
        for i, file in enumerate(files, 1):
            file_path = os.path.join(DATA_DIR, file)
            try:
                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024
                # Indicate if file was converted from PDF
                file_type = " (PDF→TXT)" if file.endswith('.txt') and any(
                    orig_name.replace('.pdf', '.txt') == file 
                    for orig_name in os.listdir(DATA_DIR) 
                    if orig_name.endswith('.pdf')
                ) else ""
                st.write(f"{i}. **{file}**{file_type} ({file_size_kb:.1f} KB)")
            except:
                st.write(f"{i}. **{file}**")
    
    # Reminder about re-indexing
    st.markdown("---")
    st.info("💡 **Note**: After deleting files, click 'Re-index Knowledgebase' to update the search index.")
    
    # Re-index button
    if st.button("🔄 Re-index Knowledgebase"):
        with st.spinner("Re-indexing..."):
            success = reindex_knowledgebase()
        if success:
            st.success("Knowledgebase re-indexed successfully!")
            st.session_state.vectorstore_loaded = True
            st.rerun()  # Refresh the page to show updated status
        else:
            st.warning("No documents found to index or indexing failed.")
            st.session_state.vectorstore_loaded = False

# --- PAGE 3: CHAT INTERFACE ---
elif page == "Chat":
    st.header("💬 Chat with Doc Brown")

    # Check if vectorstore is loaded and functional
    if 'vectorstore' not in st.session_state or not st.session_state.vectorstore_loaded:
        st.warning("⚠️ Knowledgebase is empty or failed to load. Upload files and re-index first.")

        # Show helpful info about current state
        os.makedirs(DATA_DIR, exist_ok=True)
        files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]

        if files:
            st.info(f"📁 Found {len(files)} file(s) in data directory:")
            st.info("👆 Click 'Re-index Knowledgebase' on the Upload Files page to make these files searchable.")
        else:
            st.info("📁 No files found. Go to 'Upload Files' to add documents first.")

    else:
        # Ensure we have a current chat session
        if not st.session_state.current_chat_session:
            st.session_state.current_chat_session = create_chat_session()
            st.session_state.chat_messages = []

        try:
            # Set up retriever and LLM
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            from langchain_community.llms import Bedrock
            llm = ChatBedrockConverse(
                region_name=AWS_REGION,
                model_id=AWS_BEDROCK_LLM_MODEL_ID
            )

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message_type, content, _ in st.session_state.chat_messages:
                    if message_type == "user":
                        with st.chat_message("user"):
                            st.write(content)
                    else:
                        with st.chat_message("assistant"):
                            st.write(content)

            # Chat input
            if prompt := st.chat_input("Ask Doc Brown anything..."):
                # Add user message to chat
                with st.chat_message("user"):
                    st.write(prompt)

                # Save user message to database and session state
                save_chat_message(st.session_state.current_chat_session, "user", prompt)
                st.session_state.chat_messages.append(("user", prompt, datetime.now()))
                
                # Update chat title with first user message (if this is the first message)
                if len(st.session_state.chat_messages) == 1:  # First message
                    # Create a meaningful title from the first few words
                    title_words = prompt.split()[:6]  # First 6 words
                    new_title = " ".join(title_words)
                    if len(prompt.split()) > 6:
                        new_title += "..."
                    update_chat_title(st.session_state.current_chat_session, new_title)

                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Doc is thinking..."):
                        # Retrieve relevant documents
                        docs = retriever.get_relevant_documents(prompt)
                        context = "\n\n".join([doc.page_content for doc in docs])

                        # Build prompt
                        system_prompt = f"""
You are now assuming the persona of Dr. Emmett “Doc” Brown from the Back to the Future trilogy. 
Your role is to:
- Speak and think like Doc Brown: excitable, fast-paced, brilliant, eccentric, and prone to exclamations like “Great Scott!”
- Stay consistent with Doc’s knowledge, personality, and worldview.
- Use precise technical jargon (flux capacitors, gigawatts, timelines) but explain in Doc’s quirky, animated teaching style.
- You should answer as though you don't know you're from a movie...that the storyline is your actual reality

Capabilities:
1. **Canonical QA**: When asked about Back to the Future I, II, or III, retrieve facts from the provided corpus of scripts and summarize faithfully in your own words. Use short quotes only when necessary.
2. **Speculation Beyond Canon**: If asked about “Back to the Future 4” or events after Part III, clearly label your response as speculation, theory, or invention. Maintain Doc’s voice while extrapolating logically from canon.
3. **Roleplay**: Stay in character when responding. If the user engages you in dialogue, reply as if you are Doc Brown himself, with full personality.
4. **Boundaries**: Do not reproduce large chunks of script text verbatim. Use retrieval to summarize, paraphrase, or quote briefly.

Style Guidelines:
- Always energetic and dramatic in tone.
- Use analogies, diagrams-in-words, and “mad scientist” style explanations.
- Maintain moral responsibility consistent with Doc Brown’s character (cautious about time travel’s dangers, ethical about changing history).
- Respond to user's questions conversationally, typically in a single paragraph.
- Answer any personal questions in the first person.
- Answer any questions about the storyline in the past tense as though you're telling the story your own historical knowledge

Context:
{context}
"""

                        messages = [
                            ("system", system_prompt),
                            ("human", prompt),
                        ]

                        # Generate completion
                        answer = llm.invoke(messages)
                        response_content = answer.content

                        # Display the response
                        st.write(response_content)
                        
                        # Display sources in a collapsible expander
                        if docs:
                            with st.expander(f"📚 Sources ({len(docs)} chunks retrieved)", expanded=False):
                                for i, doc in enumerate(docs, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    
                                    # Display the chunk content in a code block for better readability
                                    chunk_content = doc.page_content.strip()
                                    if len(chunk_content) > 500:
                                        # Show first 500 chars with option to expand
                                        st.markdown(f"```\n{chunk_content[:500]}...\n```")
                                        if st.button(f"Show full chunk {i}", key=f"expand_chunk_{i}"):
                                            st.markdown(f"```\n{chunk_content}\n```")
                                    else:
                                        st.markdown(f"```\n{chunk_content}\n```")
                                    
                                    if i < len(docs):  # Add separator between chunks
                                        st.markdown("---")

                        # Save assistant message to database and session state
                        save_chat_message(st.session_state.current_chat_session, "assistant", response_content)
                        st.session_state.chat_messages.append(("assistant", response_content, datetime.now()))
        except Exception as e:
            st.error(f"Error during chat: {e}")
            st.info("Please re-index your knowledgebase.")
            st.session_state.vectorstore_loaded = False
