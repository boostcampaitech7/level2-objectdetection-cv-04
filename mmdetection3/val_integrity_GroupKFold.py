import os
import json

fold = 3 # 전체 fold 수 입력
for i in range(fold):
    # Paths to JSON annotation files and image folders
    train_json_path = f'../dataset/train_fold_{i}.json'
    val_json_path = f'../dataset/val_fold_{i}.json'
    train_folder_path = '../dataset/train'

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
    # Check for overlapping images between train and val JSON files
    overlapping_ids = train_image_ids.intersection(val_image_ids) ## 이부분 뭔가 잘못
    if overlapping_ids:
        print(f"Fold {i}: There are overlapping images in train and val JSON data: {overlapping_ids} len: {len(overlapping_ids)}")
    else:
        print(f"Fold {i}: No overlapping images between train and val JSON data.")

    # Check if all JSON-specified files are present in their respective folders
    missing_in_train = train_image_files_json - train_image_files_actual
    missing_in_val = val_image_files_json - train_image_files_actual

    if missing_in_train:
        print(f"Fold {i}: The following train images are specified in JSON but missing in the folder: {missing_in_train}")
    else:
        print(f"Fold {i}: All train images specified in JSON are present in the folder.")

    if missing_in_val:
        print(f"Fold {i}: The following val images are specified in JSON but missing in the folder: {missing_in_val}")
    else:
        print(f"Fold {i}: All val images specified in JSON are present in the folder.")
