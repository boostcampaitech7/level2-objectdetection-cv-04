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
    
    #json_path = f"/data/ephemeral/home/deamin/dataset/{dataset_folder}.json"
    json_path = f"/data/ephemeral/home/data/{dataset_folder}.json"
    annotations = load_annotations(json_path)
    
    image_list = [img['file_name'] for img in annotations['images']]
    selected_image = st.selectbox("이미지를 선택하세요:", image_list)
    
    if selected_image:
        image_path = os.path.join("/data/ephemeral/home/data/", selected_image)
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
                "회전": lambda p: A.Rotate(limit=p, p=1),
                "밝기 대비 조정": lambda b, c: A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=c, p=1),
                "노이즈 추가": lambda p: A.GaussNoise(var_limit=(10 * p, 100 * p), p=1),
                "블러 효과": lambda p: A.Blur(blur_limit=int(7 * p), p=1),
                "색조 변경": lambda h, s, v: A.HueSaturationValue(hue_shift_limit=h, sat_shift_limit=s, val_shift_limit=v, p=1),
                "채도 변경": lambda p: A.RandomGamma(gamma_limit=(100 + 20 * p, 100 + 20 * p), p=1),
                "해상도 조정": lambda p: A.Resize(height=int(image.height * p), width=int(image.width * p), p=1),
                "자르기": lambda p: A.RandomCrop(height=int(image.height * p), width=int(image.width * p), p=1),
            }
            
            selected_augmentation = st.selectbox("적용할 증강 기법을 선택하세요:", 
                                         list(augmentation_options.keys()),
                                         key="augmentation_selector")
    
            augmented_image = None
            augmented_bboxes = None

            if selected_augmentation not in ["원본", "수평 뒤집기", "수직 뒤집기"]:
                augmentation = augmentation_options[selected_augmentation]
                
                if selected_augmentation in ["회전", "밝기 대비 조정", "색조 변경", "채도 변경", "노이즈 추가", "블러 효과"]:
                    if selected_augmentation == "회전":
                        angle = st.slider("회전 각도", 0, 180, 1, key="rotation_slider")
                        augmentation = augmentation(angle)
                    elif selected_augmentation == "밝기 대비 조정":
                        brightness = st.slider("밝기 조정",0.0, 1.0, 0.01, key="brightness_slider")
                        contrast = st.slider("대비 조정", 0.0, 1.0, 0.01, key="contrast_slider")
                        augmentation = augmentation(brightness, contrast)
                    elif selected_augmentation == "색조 변경":
                        hue = st.slider("색조 변경",  0, 180, 1, key="hue_slider")
                        saturation = st.slider("채도 변경",  0, 100, 1, key="saturation_slider")
                        value = st.slider("명도 변경",  0, 100, 1, key="value_slider")
                        augmentation = augmentation(hue, saturation, value)
                    elif selected_augmentation == "채도 변경":
                        gamma = st.slider("감마 값", 0.0, 2.0, 0.1, key="gamma_slider")
                        augmentation = augmentation(gamma)
                    elif selected_augmentation == "노이즈 추가":
                        strength = st.slider("증강 강도", 0, 100, 1, key="strength_slider")
                        augmentation = augmentation(strength)
                    elif selected_augmentation == "블러 효과":
                        strength = st.slider("증강 강도", 0.1, 3.0, 0.01, key="strength_slider")
                        augmentation = augmentation(strength)
                else:
                    strength = st.slider("증강 강도", 0.0, 1.0, 0.01, key="strength_slider")
                    augmentation = augmentation(strength)
                
                augmented_image, augmented_bboxes = apply_augmentation(image, bboxes, augmentation)
            elif selected_augmentation != "원본":
                augmentation = augmentation_options[selected_augmentation]
                augmented_image, augmented_bboxes = apply_augmentation(image, bboxes, augmentation)

            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="원본 이미지", use_column_width=True)
            
            with col2:
                if augmented_image is not None:
                    st.image(augmented_image, caption=f"{selected_augmentation} 적용 결과", use_column_width=True)
                else:
                    st.image(image, caption="원본 이미지", use_column_width=True)
            
            if augmented_bboxes:
                with st.expander("증강된 바운딩 박스 정보"):
                    st.write(augmented_bboxes)
    else:
        st.image(image, caption="원본 이미지", use_column_width=True)

if __name__ == '__main__':
    main()