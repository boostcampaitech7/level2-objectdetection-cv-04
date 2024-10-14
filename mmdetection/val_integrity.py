import os
import json

# Paths to JSON annotation files and image folders
train_json_path = '../dataset/train.json'
val_json_path = '../dataset/val.json'
train_folder_path = '../dataset/train'
val_folder_path = '../dataset/val'

# Load train and validation annotations
with open(train_json_path) as f:
    train_data = json.load(f)

with open(val_json_path) as f:
    val_data = json.load(f)

# Extract image IDs and file names from JSON files
train_image_ids = {img['id'] for img in train_data['images']}
val_image_ids = {img['id'] for img in val_data['images']}
train_image_files_json = {f"{str(img_id).zfill(4)}.jpg" for img_id in train_image_ids}
val_image_files_json = {f"{str(img_id).zfill(4)}.jpg" for img_id in val_image_ids}

# Get actual image files in the train and val folders
train_image_files_actual = set(os.listdir(train_folder_path))
val_image_files_actual = set(os.listdir(val_folder_path))

# Check for overlapping images between train and val JSON files
overlapping_ids = train_image_ids.intersection(val_image_ids)
if overlapping_ids:
    print("There are overlapping images in train and val JSON data:", overlapping_ids)
else:
    print("No overlapping images between train and val JSON data.")

# Check for overlapping files between train and val folders
overlapping_files = train_image_files_actual.intersection(val_image_files_actual)
if overlapping_files:
    print("There are overlapping image files in train and val folders:", overlapping_files)
else:
    print("No overlapping image files between train and val folders.")

# Check if all JSON-specified files are present in their respective folders
missing_in_train = train_image_files_json - train_image_files_actual
missing_in_val = val_image_files_json - val_image_files_actual

if missing_in_train:
    print("The following train images are specified in JSON but missing in the folder:", missing_in_train)
else:
    print("All train images specified in JSON are present in the folder.")

if missing_in_val:
    print("The following val images are specified in JSON but missing in the folder:", missing_in_val)
else:
    print("All val images specified in JSON are present in the folder.")
