import streamlit as st
import json
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import shutil
import pandas as pd
import colorsys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
# plotly 설치 필요

st.set_page_config(page_title="Object Detection Viewer", page_icon="🔍")

@st.cache_data
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_bright_colors(n):
    colors = []
    for i in range(n):
        # HSV에서 H를 균등하게 분포, S와 V는 높게 유지
        h = i / n
        s = 0.8 + random.random() * 0.2  # 0.8 ~ 1.0
        v = 0.9 + random.random() * 0.1  # 0.9 ~ 1.0
        
        # HSV를 RGB로 변환
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # RGB 값을 0-255 범위의 정수로 변환
        color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
        colors.append(color)
    return colors

@st.cache_data
def get_color_map(categories):
    color_list = generate_bright_colors(len(categories))
    return {cat['name']: color for cat, color in zip(categories, color_list)}
    
def draw_bounding_boxes(image, annotations, image_id, color_map):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
        font = ImageFont.truetype(font_path, 25)
    except IOError:
        font = ImageFont.load_default()

    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    
    for ann in annotations['annotations']:
        if ann.get('image_id') == image_id:
            bbox = ann.get('bbox')
            category_id = ann.get('category_id')
            if bbox and category_id is not None:
                category_name = categories.get(category_id, '알 수 없음')
                color = color_map.get(category_name, '#FFFFFF')
                
                # bbox가 리스트나 튜플인지 확인
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, w, h = [int(coord) for coord in bbox]
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                    text_bbox = draw.textbbox((x, y-25), category_name, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x, y-25), category_name, font=font, fill='black')
                else:
                    print(f"잘못된 bbox 형식: {bbox}")

    return image


def copy_image_to_eda(original_image_path, debug_image_path, dataset_folder, Train = True):
    # 다른 folder에 저장하고 싶으면 이 경로를 수정하세요.
    eda_folder = "../dataset/EDA/EDA_image"
    dataset_eda_folder = os.path.join(eda_folder, dataset_folder)
    os.makedirs(dataset_eda_folder, exist_ok=True)
    
    original_image_name = os.path.basename(original_image_path)
    debug_image_name = f"debug_{original_image_name}"
    
    original_destination_path = os.path.join(dataset_eda_folder, original_image_name)
    debug_destination_path = os.path.join(dataset_eda_folder, debug_image_name)
    
    copied = False
    message = ""
    
    if not os.path.exists(original_destination_path):
        shutil.copy2(original_image_path, original_destination_path)
        copied = True
        message += f"원본 이미지가 {original_destination_path}에 복사되었습니다.\n"
    else:
        message += "이미 같은 이름의 원본 이미지가 존재합니다. 복사하지 않았습니다.\n"
    
    if Train:
        if not os.path.exists(debug_destination_path):
            shutil.copy2(debug_image_path, debug_destination_path)
            copied = True
            message += f"디버그 이미지가 {debug_destination_path}에 복사되었습니다."
        else:
            message += "이미 같은 이름의 디버그 이미지가 존재합니다. 복사하지 않았습니다."
        
        return copied, message

def count_categories(annotations, image_id):
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    category_counts = {}
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            category_id = ann['category_id']
            category_name = categories.get(category_id, '알 수 없음')
            if category_name in category_counts:
                category_counts[category_name] += 1
            else:
                category_counts[category_name] = 1
    return category_counts

def main():
    st.title("Object Detection Data Viewer")

    # 이전 선택을 저장할 세션 상태 키 초기화
    if 'previous_dataset' not in st.session_state:
        st.session_state.previous_dataset = None
    if 'previous_image' not in st.session_state:
        st.session_state.previous_image = None

    dataset_folder = st.selectbox("데이터셋을 선택하세요:", ["train", "test"])
    # 데이터셋 선택이 변경되었는지 확인
    if dataset_folder != st.session_state.previous_dataset:
        st.session_state.previous_dataset = dataset_folder
        if st.session_state.previous_dataset is not None:  # 첫 실행이 아닌 경우에만 rerun
            st.rerun()   
    
    #json_path = f"/data/ephemeral/home/deamin/dataset/{dataset_folder}.json"
    json_path = f"../dataset/{dataset_folder}.json"
    annotations = load_annotations(json_path)
    
    color_map = get_color_map(annotations['categories'])
    image_list = [img['file_name'] for img in annotations['images']]
    
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("이전 이미지") and st.session_state.current_image_index > 0:
            st.session_state.current_image_index -= 1

    with col2:
        if st.button("EDA 폴더에 복사"):
            current_image = image_list[st.session_state.current_image_index]
            original_image_path = os.path.join("../dataset", current_image)
            debug_image_path = "debug_image_with_boxes.png"
            isTrain = True if dataset_folder == 'train' else 'test' 
            copied, result = copy_image_to_eda(original_image_path, debug_image_path, dataset_folder, isTrain)
            
            if copied:
                st.success("이미지가 성공적으로 복사되었습니다!")
                with st.expander("복사 세부 정보 보기"):
                    st.write(result)
            else:
                st.info("새로 복사된 이미지가 없습니다.")
                with st.expander("세부 정보 보기"):
                    st.write(result)
    with col3:
        if st.button("다음 이미지") and st.session_state.current_image_index < len(image_list) - 1:
            st.session_state.current_image_index += 1

    current_image = image_list[st.session_state.current_image_index]
    st.write(f"현재 이미지: {current_image}")

    selected_image = st.selectbox("또는 이미지를 선택하세요:", image_list, index=st.session_state.current_image_index)
    
    if selected_image != current_image:
        st.session_state.current_image_index = image_list.index(selected_image)
        current_image = selected_image
        st.rerun()

    
    if current_image:
        image_path = os.path.join("../dataset/", current_image)
        image = Image.open(image_path)

        image_info = next((img for img in annotations['images'] if img['file_name'] == current_image), None)
        if image_info:
            image_id = image_info['id']
            
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            
            image_with_boxes = draw_bounding_boxes(image, annotations, image_id, color_map)
            
            st.image(image_with_boxes, caption="바운딩 박스가 표시된 이미지", use_column_width=True)
            
            # 디버깅을 위해 이미지 저장
            image_with_boxes.save("debug_image_with_boxes.png")
            
            # 카테고리 개수 계산
            category_counts = count_categories(annotations, image_id)
            
            # 카테고리별 개수 표 생성
            st.subheader("카테고리별 어노테이션 개수")
            df = pd.DataFrame(list(category_counts.items()), columns=['카테고리', '개수'])
            df = df.sort_values('개수', ascending=False).reset_index(drop=True)
            st.table(df)


            # 파이 차트와 막대 그래프 생성
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])

            # 막대 그래프 (showlegend=False로 설정하여 범례에서 제외)
            fig.add_trace(go.Bar(x=df['카테고리'], y=df['개수'], name='카테고리별 개수',
                                text=df['개수'], textposition='auto',
                                marker_color=[color_map.get(cat, '#000000') for cat in df['카테고리']],
                                showlegend=False),
                        row=1, col=1)

            # 파이 차트
            fig.add_trace(go.Pie(labels=df['카테고리'], values=df['개수'], 
                                marker_colors=[color_map.get(cat, '#000000') for cat in df['카테고리']]),
                        row=1, col=2)

            fig.update_layout(title='카테고리별 어노테이션 분포 및 개수',
                            height=500, width=1000,
                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1))
            fig.update_xaxes(title_text='카테고리', row=1, col=1)
            fig.update_yaxes(title_text='개수', row=1, col=1)

            st.plotly_chart(fig)
            
            # 어노테이션 정보 표시
            with st.expander("Annotations INFO"):
                st.json(image_annotations)
            
        else:
            print(f"No image info found for {current_image}")
    else:
        print("No image selected")
        
if __name__ == '__main__':
    main()