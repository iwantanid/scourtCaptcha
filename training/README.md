# Training — CNN 모델 학습 모듈

PyTorch 기반 CNN(Convolutional Neural Network) 모델을 학습시키고 평가하는 모듈입니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `train.py` | CaptchaCNN 모델 정의와 학습 루프. CUDA GPU를 자동 감지하여 가속 학습을 수행한다. |
| `dataset.py` | PyTorch `Dataset` / `DataLoader` 구현체. 이미지 전처리(Grayscale, Resize, Normalize)와 실시간 데이터 증강(RandomAffine)을 담당한다. |
| `evaluate_saved.py` | 저장된 `.pth` 가중치 파일을 불러와 검증 데이터셋에 대한 정확도를 독립적으로 측정한다. |
| `answer_key.txt` | 6,090장의 캡차 이미지에 대한 정답 라벨. `파일명: 6자리숫자` 형식이다. |
| `TrainingData/` | 캡차 원본 이미지 6,090장이 들어 있는 폴더. |

---

## 환경 준비

Python 3.10 이상이 필요합니다.

```powershell
# 가상환경 (선택)
python -m venv venv
.\venv\Scripts\Activate.ps1

# PyTorch + CUDA 12.6
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126

# 기타 의존성
pip install pillow numpy
```

NVIDIA GPU가 없는 환경에서도 CPU로 학습할 수 있지만, 속도 차이가 크다.

---

## 사용법

### 모델 학습

```powershell
cd training
python -u train.py
```

*   PyTorch가 NVIDIA GPU(`cuda`)를 자동 감지하여 가속 학습을 시작한다.
*   검증 데이터 기준 6자리 완전 일치 정답률이 **99.0%**를 넘으면 즉시 학습을 멈추고, 최적의 가중치를 `captcha_model.pth`로 저장한다.
*   RTX 2070 GPU 기준 약 1분 15초 소요.

### 모델 평가

```powershell
python evaluate_saved.py
```

저장된 가중치 파일의 글자 단위 정확도와 시퀀스 단위 정확도를 출력한다.

---

## 학습 결과

| 지표 | 수치 |
|------|------|
| 글자 단위 정확도 (Character Accuracy) | **99.89%** (3,654자 중 3,650자 일치) |
| 6자리 전체 일치 정확도 (Sequence Accuracy) | **99.34%** (609개 중 605개 판독 성공) |
| 학습 에포크 | 13 에포크 (조기 종료) |
| 모델 파라미터 수 | 약 200만 개 |

---

## 데이터 라벨 노이즈 디버깅

1차 학습 시 정확도가 98.19%에서 정체되었다. 오답 11건을 시각적으로 검증한 결과, 모델의 예측이 맞고 정답지(`answer_key.txt`)의 라벨이 오기입되어 있었다. 해당 11건을 교정한 뒤 재학습하자 13 에포크 만에 99.34%를 달성했다.

이 경험은 모델 구조를 복잡하게 바꾸는 것보다 데이터 품질을 먼저 점검하는 것이 중요하다는 Data-Centric AI 관점의 좋은 사례다.
