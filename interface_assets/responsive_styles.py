import streamlit as st

def apply_responsive_config():
    """Configure page layout for responsiveness"""
    st.set_page_config(
        page_title="Cat or Dog Classifier",
        page_icon="✌️",
        layout="wide"
    )

def load_responsive_css():
    """Load and apply responsive CSS styles"""
    css = """
    <style>
    .main {
        padding: 1rem;
    }
    
    .stTitle {
        text-align: center;
        font-size: clamp(1.5rem, 4vw, 3rem);
        margin-bottom: 2rem;
    }
    
    .upload-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px dashed #cccccc;
        transition: border-color 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #1f77b4;
    }
    
    /* Responsive image styling */
    .stImage > img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Ensure images never overflow on any device */
    .stImage {
        text-align: center;
        margin: 1rem 0;
    }
    
    .responsive-image-container {
        text-align: center;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        .upload-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stTitle {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .stImage > img {
            max-width: 95% !important;
            width: auto !important;
        }
        
        .stImage {
            margin: 0.5rem 0;
        }
    }
    
    /* Tablet responsiveness */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 1.5rem;
        }
        
        .stTitle {
            font-size: 2.5rem;
        }
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def create_responsive_columns(ratios):
    """Create responsive columns with given ratios"""
    return st.columns(ratios)

def create_responsive_header(title):
    """Create a responsive header section"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<h1 class="stTitle">{title}</h1>', unsafe_allow_html=True)

def create_upload_container():
    """Create responsive upload container"""
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)

def close_upload_container():
    """Close the upload container"""
    st.markdown('</div>', unsafe_allow_html=True)

def display_files_responsive(uploaded_files):
    """Display uploaded files in a responsive grid"""
    if uploaded_files:
        st.markdown("###  Uploaded Files")
        
        # Handle both single file and multiple files
        if hasattr(uploaded_files, '__iter__') and not hasattr(uploaded_files, 'name'):
            # Multiple files (list-like)
            files_list = uploaded_files
            files_per_row = 2 if len(files_list) > 4 else 1
            
            for i in range(0, len(files_list), files_per_row):
                cols = st.columns(files_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(files_list):
                        file = files_list[i + j]
                        with col:
                            st.info(f"📄 **{file.name}**\n\nSize: {file.size / 1024:.1f} KB")
        else:
            # Single file
            file = uploaded_files
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info(f"📄 **{file.name}**\n\nSize: {file.size / 1024:.1f} KB")

def create_responsive_button(button_text, callback=None):
    """Create a responsive centered button"""
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        if st.button(button_text, use_container_width=True):
            if callback:
                return callback()
            return True
    return False

def display_responsive_image(uploaded_file, caption="Uploaded Image"):
    """Display an image with responsive sizing"""
    # Use a more mobile-friendly approach with direct width control
    col1, col2, col3 = st.columns([0.2, 2.6, 0.2])
    with col2:
        st.image(uploaded_file, caption=caption, width=None, use_column_width=True)
