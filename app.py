import streamlit as st
import warnings
import os

# Suppress warnings in production
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from prediction import predict_img
from interface_assets.responsive_styles import (
    apply_responsive_config, 
    load_responsive_css, 
    create_responsive_header,
    create_upload_container,
    close_upload_container,
    create_responsive_columns,
    display_files_responsive,
    create_responsive_button,
    display_responsive_image
)

# Apply responsive configuration
apply_responsive_config()

# Load responsive CSS
load_responsive_css()

# Create responsive header
create_responsive_header("Cat or Dog Classifier")

# Main upload section with responsive container
# create_upload_container()

# Create responsive columns for the upload area
upload_col1, upload_col2, upload_col3 = create_responsive_columns([1, 3, 1])

with upload_col2:
    uploaded_files = st.file_uploader(
        "Upload the image here",
        accept_multiple_files=False, 
        type=['jpg', 'jpeg', 'png'],
    )

close_upload_container()

# Display uploaded files in responsive layout
display_files_responsive(uploaded_files)

# Process button with responsive positioning
if uploaded_files:
    if create_responsive_button("🚀 Process Files"):
        with st.spinner("Processing files..."):
            result = predict_img(uploaded_files)
            
            # Display image with responsive sizing
            display_responsive_image(uploaded_files, "Uploaded Image")
            
            st.markdown(
                f"<div style='text-align: center; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.375rem; padding: 0.75rem; margin: 1rem 0; color: #155724;'>"
                f"<strong> The uploaded image is a {result}!</strong>"
                f"</div>", 
                unsafe_allow_html=True
            )
            # st.balloons()