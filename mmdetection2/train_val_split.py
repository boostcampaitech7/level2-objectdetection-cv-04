import json
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime

# Load annotations
annotation = '../../dataset/train.json'
with open(annotation) as f: 
    data = json.load(f)

# Prepare data for stratified group k-fold
var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']), 1))  # Dummy feature array
y = np.array([v[1] for v in var])  # Category ids
groups = np.array([v[0] for v in var])  # Image ids

# Create StratifiedGroupKFold object
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

# Use the first split for train and val
for train_idx, val_idx in cv.split(X, y, groups):
    break  # Use only the first split

# Create annotations for train and val
train_annotations = {
    'info': {
        'year': 2021,
        'version': '1.0',
        'description': 'Recycle Trash',
        'contributor': 'Upstage',
        'url': None,
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    'licenses': [{
        'id': 0,
        'name': 'CC BY 4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/deed.ast'
    }],
    'images': [],
    'annotations': [],
    'categories': data['categories']
}

val_annotations = {
    'info': {
        'year': 2021,
        'version': '1.0',
        'description': 'Recycle Trash',
        'contributor': 'Upstage',
        'url': None,
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    'licenses': [{
        'id': 0,
        'name': 'CC BY 4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/deed.ast'
    }],
    'images': [],
    'annotations': [],
    'categories': data['categories']
}

# Function to format image ID
def format_image_id(image_id):
    return f"{int(image_id):04d}.jpg"  # Format to 4 digits with leading zeros

# Fill in the train and validation data
for idx in train_idx:
    image_id = var[idx][0]
    # Append images and annotations for train set
    if not any(img['id'] == image_id for img in train_annotations['images']):
        # Get image width and height from the original annotations
        img_info = next(item for item in data['images'] if item['id'] == image_id)
        train_annotations['images'].append({
            'id': image_id,
            'file_name': format_image_id(image_id),  # Format image filename
            'width': img_info['width'],  # Add width
            'height': img_info['height'],  # Add height
            'license': 0,  # License ID
            'flickr_url': None,  # Flickr URL
            'coco_url': None,  # COCO URL
            'date_captured': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Capture date
        })
    train_annotations['annotations'].append(data['annotations'][idx])

for idx in val_idx:
    image_id = var[idx][0]
    # Append images and annotations for validation set
    if not any(img['id'] == image_id for img in val_annotations['images']):
        # Get image width and height from the original annotations
        img_info = next(item for item in data['images'] if item['id'] == image_id)
        val_annotations['images'].append({
            'id': image_id,
            'file_name': format_image_id(image_id),  # Format image filename
            'width': img_info['width'],  # Add width
            'height': img_info['height'],  # Add height
            'license': 0,  # License ID
            'flickr_url': None,  # Flickr URL
            'coco_url': None,  # COCO URL
            'date_captured': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Capture date
        })
    val_annotations['annotations'].append(data['annotations'][idx])

# Create new folders for train and val
os.makedirs('../../dataset/new_train', exist_ok=True)
os.makedirs('../../dataset/val', exist_ok=True)

# Save train and validation data to JSON files
with open('../../dataset/train.json', 'w') as f:
    json.dump(train_annotations, f)

with open('../../dataset/val.json', 'w') as f:
    json.dump(val_annotations, f)

# Move images to corresponding folders
def move_images(annotations, dest_folder):
    for img in annotations['images']:
        image_id = img['id']
        src = f'../../dataset/train/{format_image_id(image_id)}'  # Assuming original images are in 'train'
        dst = os.path.join(dest_folder, format_image_id(image_id))
        
        # Copy image to destination folder
        shutil.copy(src, dst)

# Move new train and val images to respective folders
move_images(train_annotations, '../../dataset/new_train')
move_images(val_annotations, '../../dataset/val')

# After copying, remove old train folder and rename new_train to train
shutil.rmtree('../../dataset/train', ignore_errors=True)
os.rename('../../dataset/new_train', '../../dataset/train')

print("Train and validation sets created successfully.")
