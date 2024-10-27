import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from yolov8 import YOLOv8, YOLOv8Loss

# 이미지 경로 설정 (실제 경로로 변경 필요)
IMAGE_PATH = "/data/ephemeral/home/dataset/train/0000.jpg"

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(640, 640)):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def create_dummy_target(num_objects=1):
    target = torch.zeros(1, num_objects, 6)  # [batch_size, num_objects, (x, y, w, h, class_id, obj_score)]
    for i in range(num_objects):
        target[0, i] = torch.tensor([0.5, 0.5, 0.2, 0.2, 0, 1.0])
    return target

# 모델 훈련 함수
def train_model(model, loss_fn, optimizer, image, target, num_epochs=10):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(image)
        loss, loss_items = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# 추론 함수
def inference(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    return outputs

# 메인 실행 부분
if __name__ == "__main__":
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 초기화
    num_classes = 1  # 예시로 1개 클래스 사용
    model = YOLOv8(num_classes).to(device)
    loss_fn = YOLOv8Loss(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 이미지 로드 및 전처리
    image = preprocess_image(IMAGE_PATH).to(device)
    target = create_dummy_target().to(device)

    # 모델 훈련
    print("Training model...")
    train_model(model, loss_fn, optimizer, image, target)

    # 추론
    print("Performing inference...")
    outputs = inference(model, image)

    # 결과 출력 (간단한 예시)
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape: {output.shape}")

    print("Test completed.")