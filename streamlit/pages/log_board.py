import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
@st.cache_data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df['iteration'] = df['iteration'].astype(int)
    return df

# 파일 경로 설정
file_path = 'your_file_path/level2-objectdetection-cv-04/metrics.json'
df = load_data(file_path)

# Streamlit 앱
st.title('실험 로그 분석')

# 주요 지표 선택
metrics = ['loss_box_reg', 'loss_cls', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss']
selected_metrics = st.multiselect('표시할 지표 선택:', metrics, default=['total_loss'])

# 라인 차트
st.subheader('학습 진행 상황')
fig, ax = plt.subplots(figsize=(10, 6))
for metric in selected_metrics:
    ax.plot(df['iteration'], df[metric], label=metric)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

# 최신 메트릭스
st.subheader('최신 메트릭스')
latest = df.iloc[-1]
cols = st.columns(len(selected_metrics))
for i, metric in enumerate(selected_metrics):
    cols[i].metric(metric, f"{latest[metric]:.4f}")

# 데이터 테이블
st.subheader('상세 데이터')
st.dataframe(df[['iteration'] + selected_metrics])

# 학습률 변화
st.subheader('학습률 변화')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['iteration'], df['lr'])
ax.set_xlabel('Iteration')
ax.set_ylabel('Learning Rate')
st.pyplot(fig)

# 정확도 지표
accuracy_metrics = ['fast_rcnn/cls_accuracy', 'fast_rcnn/fg_cls_accuracy']
st.subheader('정확도 지표')
fig, ax = plt.subplots(figsize=(10, 6))
for metric in accuracy_metrics:
    ax.plot(df['iteration'], df[metric], label=metric)
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)