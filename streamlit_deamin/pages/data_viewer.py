import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import shutil

st.set_page_config(page_title="Object Detection Viewer", page_icon="🔍")

@st.cache_data
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def draw_bounding_boxes(image, annotations, image_id):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    boxes_drawn = 0
    print(f"Image size: {image.size}")
    print(f"Looking for annotations with image_id: {image_id}")

    for ann in annotations['annotations']:
        if ann.get('image_id') == image_id:
            bbox = ann.get('bbox')
            if bbox:
                print(f"Drawing box: {bbox}")
                try:
                    draw.rectangle([
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    ], outline="red", width=2)
                    boxes_drawn += 1
                except Exception as e:
                    print(f"Error drawing box: {e}")
            else:
                print(f"No bbox found for annotation: {ann}")
    
    print(f"Drawn {boxes_drawn} boxes for image_id {image_id}")
    return image


def copy_image_to_eda(original_image_path, debug_image_path, dataset_folder, Train = True):
    # 다른 folder에 저장하고 싶으면 이 경로를 수정하세요.
    eda_folder = "/data/ephemeral/home/deamin/EDA_image"
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
    
    json_path = f"/data/ephemeral/home/deamin/dataset/{dataset_folder}.json"
    annotations = load_annotations(json_path)
    
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
            original_image_path = os.path.join("/data/ephemeral/home/deamin/dataset/", current_image)
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
        image_path = os.path.join("/data/ephemeral/home/deamin/dataset/", current_image)
        image = Image.open(image_path)

        image_info = next((img for img in annotations['images'] if img['file_name'] == current_image), None)
        if image_info:
            image_id = image_info['id']
            print(f"Selected image: {current_image}, Image ID: {image_id}")
            
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            print(f"Number of annotations for this image: {len(image_annotations)}")
            
            image_with_boxes = draw_bounding_boxes(image, annotations, image_id)
            
            st.image(image_with_boxes, caption="바운딩 박스가 표시된 이미지", use_column_width=True)
            
            # 디버깅을 위해 이미지 저장
            image_with_boxes.save("debug_image_with_boxes.png")
            
            # 어노테이션 정보 표시
            with st.expander("Image Annotations:"):
                    st.json(image_annotations)
            
        else:
            print(f"No image info found for {current_image}")
    else:
        print("No image selected")
        
if __name__ == '__main__':
    main()