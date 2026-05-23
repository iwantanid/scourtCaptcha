# 대법원 나의 사건 검색 Captcha Decoder

NVIDIA CUDA GPU 가속을 활용한 딥러닝(CNN) 모델로 대법원 나의 사건 검색 사이트의 6자리 숫자 캡차(CAPTCHA)를 **99.34%** 정확도로 판독하는 디코더 엔진과 웹 검증 패널입니다.

이 프로젝트는 개발자가 인공지능 에이전트와 페어 프로그래밍하며 직관과 속도에 의존하는 바이브 코딩(Vibe Coding) 기법으로 PyTorch 기반 CNN 모델 설계부터 비전 데이터 라벨링(Labeling), 최적화, FastAPI 웹 데모까지 빠르게 구축하고 훈련시키는 과정 자체에 목적을 두고 기획했습니다.

---

## 프로젝트 구조

저장소는 관심사 분리(Separation of Concerns) 원칙에 따라 모델을 연구하고 훈련시키는 training 폴더와, 모델 서빙을 담당하는 api 폴더로 나누어 구성했습니다.

```
ScourtCaptcha/ (Root)
│
├── README.md               # 프로젝트 전체 사용 설명서 및 면책 조항
├── architecture_guide.md   # 대학 학부생 대상의 딥러닝/CNN 동작 원리 해설서
│
├── training/               # 1. 모델 연구 및 학습 모듈
│   ├── dataset.py          # PyTorch 데이터 로더 & 실시간 데이터 증강(Augmentation)
│   ├── train.py            # CNN 모델 아키텍처 및 훈련 루프 스크립트
│   ├── evaluate_saved.py   # 저장된 가중치 검증 평가 스크립트
│   ├── answer_key.txt      # 6,090개의 정제된 정답 라벨 (0000.png ~ 6089.png)
│   └── TrainingData/       # 6,090장의 캡차 학습용 원본 이미지 폴더
│
└── api/                    # 2. 모델 배포 및 서비스 API 모듈
    ├── app.py              # FastAPI 서버와 드래그 앤 드롭 웹 검증 패널
    └── captcha_model.pth   # 학습 완료된 최적의 모델 가중치 파일 (99.34% 정확도)
```

> **상대 경로 기반 설계**: 모든 스크립트는 상대 경로(Relative Path)를 사용합니다. 따라서 깃허브에서 저장소를 클론한 뒤 절대 경로를 수정할 필요 없이, 터미널 명령어 한 줄로 바로 학습하고 실행할 수 있습니다.

---

## 설치 및 환경 구성

이 프로젝트는 Windows OS와 Python 3.13 환경에 맞춰 개발했습니다. NVIDIA GPU가 탑재된 컴퓨터에서는 CUDA 가속을 지원하므로 학습을 훨씬 빠르게 마칠 수 있습니다.

### 1. 가상환경 생성 및 활성화 (선택)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. PyTorch 및 CUDA 12.6 가속 라이브러리 설치
NVIDIA GPU로 학습 속도를 높이려면 CUDA를 지원하는 PyTorch 패키지를 설치해야 합니다.
```powershell
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126
```

### 3. FastAPI 및 기타 의존성 라이브러리 설치
```powershell
pip install fastapi uvicorn pillow numpy
```

---

## 사용 방법

### 1. 모델 재학습 (Training)
라벨 오류를 교정한 6,090장의 이미지 데이터셋과 정답지를 바탕으로 CNN 모델을 처음부터 다시 학습시킵니다.
```powershell
cd training
python -u train.py
```
*   **작동 방식**: PyTorch가 시스템에 장착된 NVIDIA GPU(`cuda`)를 자동으로 찾아서 하드웨어 가속 학습을 시작합니다.
*   **조기 종료 기능**: 검증 데이터셋 기준 6자리 완전 일치 정답률이 **99.0%**를 넘어서는 즉시 훈련을 중단하고, 최적의 가중치를 `captcha_model.pth` 파일로 저장합니다. (RTX 2070 GPU 기준 약 1분 15초 소요)

### 2. 학습 모델 성능 평가
저장해 둔 모델 가중치의 오차율과 정확도를 독립적으로 정밀하게 검증합니다.
```powershell
cd training
python evaluate_saved.py
```

### 3. API 및 웹 검증 패널 구동 (Serving)
학습한 모델 가중치를 사용해 글래스모피즘(Glassmorphism) 디자인의 드래그 앤 드롭 웹 검증 화면을 구동합니다.
```powershell
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*   서버를 구동하고 브라우저에서 `http://localhost:8000`에 접속하면, 화면에 준비된 테스트용 캡차를 클릭해 보거나 직접 준비한 캡차 이미지를 끌어다 놓아 판독 결과를 즉시 확인할 수 있습니다.

### 4. API 호출 가이드 (HTTP POST /predict)
FastAPI 서버가 제공하는 `/predict` 엔드포인트를 사용해 프로그램이나 웹에서 직접 캡차 판독을 요청하고 응답받을 수 있습니다. 요청 데이터는 `multipart/form-data` 형식으로 전송해야 하며, 키값은 `file`입니다.

#### curl을 통한 터미널 요청
터미널에서 `curl` 명령어로 로컬 캡차 이미지를 전송하여 결과를 즉시 수신합니다.
```bash
curl -X POST -F "file=@captcha_sample.png" http://localhost:8000/predict
```
*   **응답 JSON 구조**
    ```json
    {
      "status": "success",
      "prediction": "629474"
    }
    ```

#### JavaScript (Fetch API) 요청 예시
웹 브라우저나 프론트엔드 환경에서 비동기로 캡차 이미지를 서버로 전송하고 결과를 가로채는 자바스크립트 구현체 예시입니다.
```javascript
const formData = new FormData();
// fileInput은 <input type="file"> 요소 또는 이미지 File 객체
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.status === 'success') {
    console.log('판독된 6자리 숫자:', data.prediction); // 예: "629474"
  } else {
    console.error('판독 연산 실패');
  }
})
.catch(error => console.error('네트워크 연결 에러:', error));
```

#### Python (Requests 라이브러리) 요청 예시
다른 자동화 프로그램이나 백엔드 파이썬 코드에서 캡차 판독 기능을 연동할 때 사용하는 요청 예시입니다.
```python
import requests

url = "http://localhost:8000/predict"
image_path = "captcha_sample.png"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print("판독 결과:", result["prediction"])  # 예: "629474"
else:
    print("API 호출 오류:", response.text)
```

---

## 모델 성능 실측치

*   **글자 단위 정확도 (Character Accuracy)**: **99.89%** (3,654글자 중 3,650글자 일치, 4글자 오답)
*   **6자리 전체 일치 정확도 (Validation Sequence Accuracy)**: **99.34%** (609개 검증용 캡차 중 605개 판독 성공)
*   **이미지 1장 판독 속도**: CPU 기준 약 **5ms** (0.005초), GPU 기준 **1ms** 내외

---

## 면책 조항 (Disclaimer)

1.  **사용 권한 및 목적**: 이 프로그램은 오직 기술 연구, 딥러닝 학습 및 개인적/상업적 검증 목적으로만 이용할 수 있습니다. 본 프로그램을 악의적으로 활용해 사법부나 국가 기관 시스템에 자동화된 봇(Bot)으로 접근하거나 비정상적인 트래픽을 유발하는 행위는 엄격히 금지합니다. 이를 위반하여 발생하는 모든 책임은 전적으로 프로그램을 실행한 사용자에게 있습니다.
2.  **관계 당국 권고 수용 및 비공개 조치**: 대한민국 법률이나 대법원 사이트 보안 규정에 저촉된다는 법원 관계자 혹은 시스템 관리자의 정당한 권고나 연락을 받으면, 이 저장소는 어떠한 이의 없이 즉시 **비공개(Private Repository)**로 전환합니다.
