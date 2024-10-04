import streamlit as st
import albumentations as A
import cv2
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(page_title="data argm tester", page_icon="🔍")
st.title('데이터 증강 테스터')

@st.cache_data
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def apply_augmentation(image, bboxes, augmentation):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    augmented = augmentation(image=image_bgr, bboxes=bboxes)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    
    augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(augmented_image_rgb), augmented_bboxes

def main():
    dataset_folder = st.selectbox("데이터셋을 선택하세요:", ["train", "test"])
    
    json_path = f"/data/ephemeral/home/deamin/dataset/{dataset_folder}.json"
    annotations = load_annotations(json_path)
    
    image_list = [img['file_name'] for img in annotations['images']]
    selected_image = st.selectbox("이미지를 선택하세요:", image_list)
    
    if selected_image:
        image_path = os.path.join("/data/ephemeral/home/deamin/dataset/", selected_image)
        image = Image.open(image_path)
        
        image_info = next((img for img in annotations['images'] if img['file_name'] == selected_image), None)
        if image_info:
            image_id = image_info['id']
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            
            bboxes = [[ann['bbox'][0], ann['bbox'][1], 
                       ann['bbox'][0] + ann['bbox'][2], 
                       ann['bbox'][1] + ann['bbox'][3], 
                       ann['category_id']] for ann in image_annotations]
            
            augmentation_options = {
                "원본": None,
                "수평 뒤집기": A.HorizontalFlip(p=1),
                "수직 뒤집기": A.VerticalFlip(p=1),
                "회전": A.Rotate(limit=45, p=1),
                "밝기 대비 조정": A.RandomBrightnessContrast(p=1),
                "노이즈 추가": A.GaussNoise(p=1),
                "블러 효과": A.Blur(blur_limit=7, p=1),
                "색조 변경": A.HueSaturationValue(p=1),
                "채도 변경": A.RandomGamma(p=1),
                "크기 조정": A.Resize(height=int(image.height*0.8), width=int(image.width*0.8), p=1),
                "자르기": A.RandomCrop(height=int(image.height*0.8), width=int(image.width*0.8), p=1),
            }
            
            selected_augmentation = st.selectbox("적용할 증강 기법을 선택하세요:", list(augmentation_options.keys()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="원본 이미지", use_column_width=True)
            
            with col2:
                if selected_augmentation != "원본":
                    augmentation = augmentation_options[selected_augmentation]
                    augmented_image, augmented_bboxes = apply_augmentation(image, bboxes, augmentation)
                    st.image(augmented_image, caption=f"{selected_augmentation} 적용 결과", use_column_width=True)
                    
                    with st.expander("증강된 바운딩 박스 정보"):
                        st.write(augmented_bboxes)
                else:
                    st.image(image, caption="원본 이미지", use_column_width=True)

if __name__ == '__main__':
    main()