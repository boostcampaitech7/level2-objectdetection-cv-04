import streamlit as st

st.set_page_config(page_title="러닝메이트 pj2 팀 페이지", page_icon="🧊")

st.write("# 러닝메이트 Object Detection:running:")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is our team's main page for object detection.
    Use the menu on the left to navigate to different pages.
    
    ### Pages:
    - Object Detection Data Viewer
    - Second Page
    - Third Page
    """
)