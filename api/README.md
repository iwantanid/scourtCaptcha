# API — 캡차 판독 서버 및 웹 패널

학습이 끝난 CNN 모델 가중치를 FastAPI로 서빙하는 모듈이다. 캡차 이미지를 HTTP POST로 보내면 6자리 숫자를 JSON으로 돌려준다. 브라우저에서 접속하면 드래그 앤 드롭 방식의 웹 검증 패널도 사용할 수 있다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `app.py` | FastAPI 서버. 모델 로딩, `/predict` 엔드포인트, 웹 UI를 모두 포함한다. |
| `captcha_model.pth` | 학습 완료된 최적의 가중치 파일 (99.34% 정확도). |

---

## 환경 준비

```powershell
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126
pip install fastapi uvicorn pillow numpy
```

---

## 서버 실행

```powershell
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

서버가 뜨면 브라우저에서 `http://localhost:8000`에 접속한다. 화면에 표시된 샘플 캡차를 클릭하거나, 직접 준비한 캡차 이미지를 끌어다 놓으면 판독 결과가 바로 나온다.

---

## API 명세

### `POST /predict`

캡차 이미지 파일을 받아 6자리 숫자를 예측한다.

**요청**

*   Content-Type: `multipart/form-data`
*   필드명: `file` (PNG, JPG 등 이미지 파일)

**응답** (JSON)

```json
{
  "status": "success",
  "prediction": "629474"
}
```

에러가 발생하면 HTTP 400과 함께 `detail` 필드에 원인이 담긴다.

---

## 호출 예시

### curl

```bash
curl -X POST -F "file=@captcha.png" http://localhost:8000/predict
```

### Python

```python
import requests

with open("captcha.png", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"file": f})

print(resp.json()["prediction"])  # "629474"
```

### JavaScript (Fetch API)

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: form,
});
const data = await res.json();
console.log(data.prediction); // "629474"
```

---

## 판독 성능

| 지표 | 수치 |
|------|------|
| 이미지 1장 판독 속도 (CPU) | 약 5ms |
| 이미지 1장 판독 속도 (GPU) | 약 1ms |
