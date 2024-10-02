import streamlit as st
import json
import os
from PIL import Image, ImageDraw

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

def main():
    st.title("Object Detection Data Viewer")

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
            print(f"Selected image: {selected_image}, Image ID: {image_id}")
            
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            print(f"Number of annotations for this image: {len(image_annotations)}")
            
            image_with_boxes = draw_bounding_boxes(image, annotations, image_id)
            
            st.image(image_with_boxes, caption="바운딩 박스가 표시된 이미지", use_column_width=True)
            
            # 디버깅을 위해 이미지 저장
            image_with_boxes.save("debug_image_with_boxes.png")
            
            # 어노테이션 정보 표시
            st.write("Image Annotations:")
            st.json(image_annotations)
        else:
            print(f"No image info found for {selected_image}")
    else:
        print("No image selected")

if __name__ == "__main__":
    main()