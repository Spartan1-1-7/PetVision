import streamlit as st
from prediction import predict_img
from interface_assets.responsive_styles import (
    apply_responsive_config, 
    load_responsive_css, 
    create_responsive_header,
    create_upload_container,
    close_upload_container,
    create_responsive_columns,
    display_files_responsive,
    create_responsive_button
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
            st.success(f"The uploaded image is a {result}!")
            # st.balloons()