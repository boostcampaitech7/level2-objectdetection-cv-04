## Image Object detection competition

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## Data

Input : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 됩니다. bbox annotation은 COCO format으로 제공됩니다. (COCO format에 대한 설명은 학습 데이터 개요를 참고해주세요.)

Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요.)


## Model 

Baseline code는 Faster R-CNN을 기본값으로 두고 있습니다. 
detectron2는 최신 버전까지, mmdetection은 2.x 버전까지 지원하므로 모델 사용에 제한이 있습니다.

## Usage

### Installation

0. setup
   ```
   apt update
   apt upgrade
   apt-get update -y
   apt-get install -y libgl1-mesa-glx
   apt-get install -y libglib2.0-0
   apt install wget
   python -m pip install --upgrade pip
   pip install ninja
   ```
   
2. Clone the repository & download data:
   ```
   git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-04.git
   cd level2-objectdetection-cv-04.git
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Unzip data
   ```
   cd data
   wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz
   tar -zxvf data.tar.gz
   ```

### Training(수정 필요)

```
python train.py --traindata_dir ./data/train --traindata_info_file ./data/train.csv --save_result_path ./train_result --log_dir ./logs --val_split 0.2 --transform_type albumentations --batch_size 64 --model_type timm --model_name eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --pretrained True --learning_rate 0.001 --epochs_per_lr_decay 2 --scheduler_gamma 0.1 --epochs 5
```

hyperparameter는 원하는 만큼 수정할 수 있습니다.

### Inference


```
python inference.py
```

## Project Structure
```
project_root/
│
├── dataset/
│   ├── test/
│   ├── train/
│   ├── test.json
│   └── train.json
│
├── detectron2/
│   ├── detectron2 folders
│   ├── train.py
│   └── inference.py
│
├── faster_rcnn/
│   ├── train.py
│   └── inference.py
│
├── mmdetection/
│   ├── mmdetection folders
│   ├── train.py
│   └── inference.py
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── trainer.py
│   ├── layer_modification
│   └── utils.py
└── README.md
```


----
## Experiment (~24.09.20)
### Experiment 규칙
<br>
Step1 .  반드시 Kanban에 실험 계획을 올려주세요
<br>
  Step2. 실험 후 실험 내용은 브랜치 명 "exp/{실험할 내용}"으로 깃에 올려주세요.<br>
 Step3.  실험 결과는 구글 시트에 기록해주세요

### Hyperparameter Tuning Experiment
- Write in [Google sheet](https://docs.google.com/spreadsheets/d/1tuTotQ_ALJQyJPzXt2NMeeyWfkm5csweRrYfWxnff8A/edit?usp=sharing)



### 브랜치 작성 규칙
1. main 브랜치는 건들지 말아주세요
2. feature 관련 브랜치명은 "feat/{구현할 내용}".
3. 각종 실험 관련 브랜치명은 "exp/{실험할내용}".
4. 수정 사항 관련 브랜치명은 "fix/{수정할 내용}"
   
