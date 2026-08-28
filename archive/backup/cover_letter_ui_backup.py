# Backup of cover letter UI and related behavior from app.py
# Moved here when 'narrow to CV only' feature was applied.

import streamlit as st
from src.document_parser import parse_uploaded_file

# Original cover letter UI block
st.markdown("**Cover Letter**")
uploaded_cl = st.file_uploader("Upload cover letter (TXT, DOCX, PDF)", type=["txt", "pdf", "docx"], key="cl_upload")
paste_cl = st.text_area("Or paste cover letter text (used if no file uploaded)")

cl_text, cl_err = parse_uploaded_file(uploaded_cl)
if cl_err:
    st.error(f"Cover letter upload: {cl_err}")

final_cl = cl_text if cl_text else paste_cl

# Note: the live app now passes an empty string for cover_letter to provider.analyse
# to focus the UI and features on CV only.
