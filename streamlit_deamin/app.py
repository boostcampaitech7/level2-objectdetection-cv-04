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

# 링크와 버튼 텍스트를 딕셔너리로 정의
links = {
    "깃허브 확인하러 가기": "https://github.com/boostcampaitech7/level2-objectdetection-cv-04",
    "AI 스테이지 제출하러 가기": "https://stages.ai/",
    "Notion Log 관리하러 가기": "https://rare-ursinia-b60.notion.site/log-ee03a232d1994ebfa06f5ae4b207e2e0",
    "Log excel Sheet 직접 보러가기": "https://docs.google.com/spreadsheets/d/1Dre5HGZEvnyhmMDjIxTU_AtzM_A2Fm6wBpROrXat7qw/edit?gid=0#gid=0",
    "부스트캠프 바로가기": "https://www.boostcourse.org/boostcampaitech7",
    "Zoom 회의실 바로가기": "https://us06web.zoom.us/j/89925368049?pwd=CBdo2IbuOsqAI31QEmEt2VE2fFc73W.1"
}

# 각 링크에 대해 버튼 생성
for text, url in links.items():
    if st.button(text):
        st.markdown(f'<a href="{url}" target="_blank">{text}</a>', unsafe_allow_html=True)
