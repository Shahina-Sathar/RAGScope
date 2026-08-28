"""Documents page - Upload and manage documents."""
import streamlit as st
from utils.api_client import RAGScopeAPI
import tempfile
import os


def render_documents(api_client: RAGScopeAPI):
    """Render documents page."""
    st.markdown('<p class="main-header">📄 Documents</p>', unsafe_allow_html=True)
    st.markdown("Upload and manage your PDF documents")
    st.markdown("---")
    
    # Check API connection
    if not api_client.health_check():
        st.error("❌ Backend API is not connected")
        return
    
    # Upload section
    st.subheader("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to be indexed and used for RAG queries"
    )
    
    if uploaded_file is not None:
        st.info(f"Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        if st.button("📤 Upload and Index", type="primary"):
            with st.spinner("Uploading document..."):
                try:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Upload document
                    result = api_client.upload_document(
                        tmp_path, filename=uploaded_file.name
                    )
                    document_id = result.get("id")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Document uploaded successfully!")
                    
                    # Index document
                    with st.spinner("Indexing document..."):
                        index_result = api_client.index_document(document_id)
                        st.success(f"✅ Document indexed! {index_result.get('num_chunks', 0)} chunks created.")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error uploading document: {e}")
    
    st.markdown("---")
    
    # Document list
    st.subheader("📚 Document Library")
    
    try:
        documents = api_client.list_documents()
        
        if not documents:
            st.info("No documents uploaded yet. Upload a PDF to get started!")
            return
        
        # Display documents in a table
        for doc in documents:
            with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**ID:** `{doc.get('id', 'N/A')}`")
                    st.markdown(f"**Pages:** {doc.get('num_pages', 0)}")
                
                with col2:
                    st.markdown(f"**Chunks:** {doc.get('num_chunks', 0)}")
                    status = doc.get("status", "unknown")
                    status_emoji = {
                        "uploaded": "📤",
                        "indexed": "✅",
                        "failed": "❌",
                    }.get(status, "❓")
                    st.markdown(f"**Status:** {status_emoji} {status}")
                
                with col3:
                    st.markdown(f"**Created:** {doc.get('created_at', 'N/A')[:19]}")
                    if doc.get("indexed_at"):
                        st.markdown(f"**Indexed:** {doc.get('indexed_at')[:19]}")
                
                # Delete button
                if st.button("🗑️ Delete", key=f"delete_{doc.get('id')}"):
                    try:
                        api_client.delete_document(doc.get("id"))
                        st.success("Document deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting document: {e}")
        
        # Summary
        st.markdown("---")
        total_docs = len(documents)
        total_pages = sum(d.get("num_pages", 0) for d in documents)
        total_chunks = sum(d.get("num_chunks", 0) for d in documents)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", total_docs)
        with col2:
            st.metric("Total Pages", total_pages)
        with col3:
            st.metric("Total Chunks", total_chunks)
    
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
