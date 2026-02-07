from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
import os
from typing import List, Dict
import json
import io


def process_pdf(pdf_file) -> tuple:
    """Extract text from a single PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    return text, pdf_file.name


def merge_pdfs(pdf_files) -> bytes:
    """Merge multiple PDFs into one"""
    pdf_writer = PdfWriter()
    
    for pdf_file in pdf_files:
        pdf_file.seek(0)  # Reset file pointer
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
    
    # Write to bytes
    output = io.BytesIO()
    pdf_writer.write(output)
    output.seek(0)
    return output.getvalue()


def create_vector_store(pdf_texts: Dict[str, str]):
    """Create FAISS vector store from PDF texts"""
    documents = []
    
    for pdf_name, text in pdf_texts.items():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_name,
                    "chunk_id": i
                }
            )
            documents.append(doc)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore, documents


def get_relevant_context(vectorstore, query: str, k: int = 5) -> tuple:
    """Retrieve relevant chunks from vector store"""
    docs = vectorstore.similarity_search(query, k=k)
    
    # Group by source PDF
    context_by_source = {}
    for doc in docs:
        source = doc.metadata['source']
        if source not in context_by_source:
            context_by_source[source] = []
        context_by_source[source].append(doc.page_content)
    
    return docs, context_by_source


def generate_response(client: Groq, query: str, context_by_source: Dict, task_type: str = "answer"):
    """Generate response using Groq API"""
    
    # Build context string with source information
    context_parts = []
    for source, chunks in context_by_source.items():
        context_parts.append(f"### From {source}:")
        context_parts.append("\n".join(chunks))
        context_parts.append("")
    
    full_context = "\n".join(context_parts)
    
    # Different system prompts based on task type
    system_prompts = {
        "answer": """You are a helpful assistant that answers questions based on provided PDF documents. 
- Give direct, accurate answers using information from the context
- When information comes from multiple PDFs, mention the sources
- If comparing PDFs, clearly highlight similarities and differences
- If information is not in the documents, say so clearly""",
        
        "email": """You are a professional email writer. Based on the document context provided:
- Write a clear, professional email
- Use appropriate email format (Subject, Greeting, Body, Closing)
- Incorporate relevant information from the documents
- Keep it concise and well-structured""",
        
        "summary": """You are an expert at summarizing documents. 
- Create a comprehensive summary of the key points
- If multiple documents are provided, organize by source or theme
- Highlight important findings, data, or conclusions
- Be concise but thorough""",
        
        "comparison": """You are an expert at comparing and analyzing multiple documents.
- Compare ALL uploaded documents systematically
- Identify key similarities and differences between each document
- Organize your comparison in a clear structure with sections for each document
- Highlight unique points from each document
- Provide insights on the overall comparison across all documents
- Create a comprehensive comparison table if helpful""",
        
        "merge": """You are an expert at merging multiple documents into one cohesive document.
- Combine information from ALL uploaded PDFs
- Organize the merged content logically by topics or themes
- Remove duplicate information
- Maintain all important details from each source
- Clearly indicate which information came from which source
- Create a well-structured, comprehensive merged document"""
    }
    
    system_prompt = system_prompts.get(task_type, system_prompts["answer"])
    
    user_prompt = f"""Context from uploaded PDF documents:

{full_context}

User Request: {query}

Please provide your response:"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3 if task_type == "answer" else 0.7,
            max_tokens=3000
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def detect_task_type(query: str) -> str:
    """Detect what type of task the user is requesting"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["merge", "combine", "join", "merge pdfs", "combine pdfs"]):
        return "merge"
    elif any(word in query_lower for word in ["email", "write an email", "draft an email", "compose email"]):
        return "email"
    elif any(word in query_lower for word in ["summarize", "summary", "give me a summary"]):
        return "summary"
    elif any(word in query_lower for word in ["compare", "comparison", "difference", "similar", "contrast"]):
        return "comparison"
    else:
        return "answer"


def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Multi-PDF AI Assistant", 
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Multi-PDF AI Assistant")
    st.markdown("Upload multiple PDFs and ask questions, compare documents, merge PDFs, or generate emails!")
    
    # Initialize session state
    if 'pdf_texts' not in st.session_state:
        st.session_state['pdf_texts'] = {}
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'uploaded_files_list' not in st.session_state:
        st.session_state['uploaded_files_list'] = []
    
    # API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.stop()
    
    client = Groq(api_key=api_key)
    
    # Sidebar for PDF management
    with st.sidebar:
        st.header("📁 PDF Management")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            # Show number of uploaded files
            st.info(f"📊 {len(uploaded_files)} PDF file(s) uploaded")
            
            if st.button("🔄 Process PDFs", type="primary"):
                with st.spinner("Processing PDFs..."):
                    st.session_state['pdf_texts'] = {}
                    st.session_state['uploaded_files_list'] = []
                    
                    for pdf_file in uploaded_files:
                        text, filename = process_pdf(pdf_file)
                        st.session_state['pdf_texts'][filename] = text
                        st.session_state['uploaded_files_list'].append(pdf_file)
                    
                    # Create vector store
                    vectorstore, documents = create_vector_store(st.session_state['pdf_texts'])
                    st.session_state['vectorstore'] = vectorstore
                    
                    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully!")
                    st.balloons()
        
        # Display loaded PDFs
        if st.session_state['pdf_texts']:
            st.subheader(f"📄 Loaded PDFs ({len(st.session_state['pdf_texts'])}):")
            for i, pdf_name in enumerate(st.session_state['pdf_texts'].keys(), 1):
                st.write(f"{i}. {pdf_name}")
            
            # Merge PDFs option
            st.divider()
            if len(st.session_state['pdf_texts']) > 1:
                if st.button("🔗 Download Merged PDF", type="secondary"):
                    with st.spinner("Merging PDFs..."):
                        merged_pdf = merge_pdfs(st.session_state['uploaded_files_list'])
                        st.download_button(
                            label="⬇️ Download Merged PDF",
                            data=merged_pdf,
                            file_name="merged_document.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDFs merged successfully!")
        
        # Clear all button
        if st.session_state['pdf_texts']:
            st.divider()
            if st.button("🗑️ Clear All PDFs", type="secondary"):
                st.session_state['pdf_texts'] = {}
                st.session_state['vectorstore'] = None
                st.session_state['chat_history'] = []
                st.session_state['uploaded_files_list'] = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state['pdf_texts']:
        st.info("👈 Please upload PDF files using the sidebar to get started!")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Multi-PDF Support")
            st.write("Upload and analyze multiple PDFs simultaneously")
        with col2:
            st.markdown("### 🔍 Smart Comparison")
            st.write("Compare content across all your documents")
        with col3:
            st.markdown("### 🔗 Merge PDFs")
            st.write("Combine multiple PDFs into one file")
    else:
        # Task type selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_input(
                "💬 Ask a question or request a task:",
                placeholder="e.g., 'Compare all PDFs' or 'Merge the content from all documents'"
            )
        
        with col2:
            task_type = st.selectbox(
                "Task Type",
                ["Auto-detect", "Answer", "Email", "Summary", "Comparison", "Merge Content"],
                help="Select task type or let AI auto-detect"
            )
        
        # Example queries
        with st.expander("💡 Example queries"):
            st.markdown(f"""
            **Questions (Working with {len(st.session_state['pdf_texts'])} PDFs):**
            - What are the main findings across all documents?
            - Compare the methodology used in each PDF
            - What are the differences between all uploaded documents?
            
            **Merge & Combine:**
            - Merge all PDFs into one comprehensive document
            - Combine the key points from all documents
            - Create a unified summary of all PDFs
            
            **Email Generation:**
            - Write an email summarizing the key points from all documents
            - Draft an email comparing findings across all PDFs
            
            **Summaries:**
            - Summarize each PDF separately
            - Give me a combined summary of all documents
            """)
        
        if user_question and st.session_state['vectorstore']:
            with st.spinner("🤖 Processing your request..."):
                # Detect or use selected task type
                if task_type == "Auto-detect":
                    detected_task = detect_task_type(user_question)
                elif task_type == "Merge Content":
                    detected_task = "merge"
                else:
                    detected_task = task_type.lower()
                
                # Get relevant context - increase k for multiple PDFs
                num_pdfs = len(st.session_state['pdf_texts'])
                k_value = min(10, num_pdfs * 3)  # Get more chunks for multiple PDFs
                
                docs, context_by_source = get_relevant_context(
                    st.session_state['vectorstore'],
                    user_question,
                    k=k_value
                )
                
                # Generate response
                response = generate_response(
                    client,
                    user_question,
                    context_by_source,
                    detected_task
                )
                
                # Display response
                st.subheader(f"📝 Response ({detected_task.title()}) - Analyzed {len(context_by_source)} document(s):")
                st.markdown(response)
                
                # Show sources
                with st.expander(f"🔍 View source documents and context ({len(context_by_source)} sources)"):
                    for source, chunks in context_by_source.items():
                        st.markdown(f"**📄 {source}:** ({len(chunks)} chunk(s))")
                        for i, chunk in enumerate(chunks, 1):
                            st.text_area(
                                f"Chunk {i}",
                                chunk,
                                height=100,
                                key=f"{source}_{i}_{len(st.session_state['chat_history'])}"
                            )
                        st.divider()
                
                
                st.session_state['chat_history'].append({
                    'question': user_question,
                    'answer': response,
                    'task_type': detected_task,
                    'sources': list(context_by_source.keys()),
                    'num_sources': len(context_by_source)
                })
        
        
        if st.session_state['chat_history']:
            with st.expander(f"📜 Chat History ({len(st.session_state['chat_history'])} conversations)", expanded=False):
                for i, chat in enumerate(reversed(st.session_state['chat_history']), 1):
                    st.markdown(f"**Q{i} ({chat['task_type']}) - {chat['num_sources']} sources:** {chat['question']}")
                    st.markdown(f"**A{i}:** {chat['answer'][:300]}...")
                    st.markdown(f"*Sources: {', '.join(chat['sources'])}*")
                    st.divider()


if __name__ == "__main__":
    main()

### python -m streamlit run app.py