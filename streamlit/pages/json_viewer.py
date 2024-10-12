import streamlit as st
import json
import pandas as pd

@st.cache_data
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    st.title("JSON 파일 형식 뷰어")

    dataset_folder = st.selectbox("데이터셋을 선택하세요:", ["train", "test"])
    
    #json_path = f"/data/ephemeral/home/deamin/dataset/{dataset_folder}.json"
    json_path = f"/data/ephemeral/home/data/{dataset_folder}.json"
    annotations = load_annotations(json_path)
    
    image_list = [img['file_name'] for img in annotations['images']]
    
    selected_image = st.selectbox("이미지를 선택하세요:", image_list)

    if selected_image:
        image_info = next((img for img in annotations['images'] if img['file_name'] == selected_image), None)
        if image_info:
            image_id = image_info['id']
            
            # 카테고리 정보 딕셔너리 생성
            categories = {cat['id']: cat['name'] for cat in annotations['categories']}
            
            # 선택된 이미지에 대한 어노테이션 추출 및 카테고리 이름 추가
            image_annotations = []
            category_counts = {}
            for ann in annotations['annotations']:
                if ann['image_id'] == image_id:
                    ann_copy = ann.copy()
                    category_id = ann['category_id']
                    category_name = categories.get(category_id, '알 수 없음')
                    ann_copy['category_name'] = category_name
                    image_annotations.append(ann_copy)
                    
                    # 카테고리별 개수 세기
                    if category_name in category_counts:
                        category_counts[category_name] += 1
                    else:
                        category_counts[category_name] = 1
            # 카테고리별 개수 표 생성
            st.subheader("카테고리별 어노테이션 개수")
            df = pd.DataFrame(list(category_counts.items()), columns=['카테고리', '개수'])
            df = df.sort_values('개수', ascending=False).reset_index(drop=True)
            st.table(df)
           
            # 이미지 정보 표시
            with st.expander("이미지 정보"):
                st.json(image_info)
            
            # 어노테이션 정보 표시
            with st.expander("어노테이션 정보"):
                st.json(image_annotations)
            
            # 카테고리 정보 표시
            with st.expander("카테고리 정보"):
                st.json(annotations['categories'])

        else:
            st.error("선택한 이미지에 대한 정보를 찾을 수 없습니다.")
if __name__ == "__main__":
    main()