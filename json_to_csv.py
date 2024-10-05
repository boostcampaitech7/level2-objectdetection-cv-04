import json
import csv

def json_to_csv(json_file, images_csv, annotations_csv):
    # JSON 파일 열기
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # images 리스트 추출
    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # images 데이터를 CSV로 변환
    with open(images_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if images:
            header = images[0].keys()
            writer.writerow(header)
            for row in images:
                writer.writerow(row.values())
        else:
            print("images 리스트가 비어 있습니다.")

    # annotations 데이터를 CSV로 변환
    with open(annotations_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if annotations:
            header = annotations[0].keys()
            writer.writerow(header)
            for row in annotations:
                writer.writerow(row.values())
        else:
            print("annotations 리스트가 비어 있습니다.")

# 파일 경로 설정
train_json_file = 'dataset/train.json'
test_json_file = 'dataset/test.json'
train_img_csv_file = 'train_image.csv'
train_anno_csv_file = 'train_annotation.csv'
test_img_csv_file = 'test_image.csv'
test_anno_csv_file = 'test_annotation.csv'

# 함수 호출
json_to_csv(train_json_file, train_img_csv_file, train_anno_csv_file)
json_to_csv(test_json_file, test_img_csv_file, test_anno_csv_file)
