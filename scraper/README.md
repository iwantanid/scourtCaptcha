# Captcha Image Scraper (캡차 이미지 수집기)

대법원 나의 사건 검색 사이트(`scourt.go.kr`)에서 캡차(CAPTCHA) 이미지를 자동으로 대량 수집하는 Node.js 스크래퍼입니다.

수집한 이미지는 `training/TrainingData/` 폴더에 옮겨 CNN 모델의 학습 데이터로 사용합니다.

---

## 기술 배경

이 스크래퍼는 **Playwright** 브라우저 자동화 라이브러리로 동작합니다. 대법원 사이트의 캡차는 단순 `<img>` 태그가 아니라 iframe 안에 동적으로 렌더링되기 때문에, 일반적인 HTTP 요청(`fetch`, `axios` 등)으로는 받아올 수 없습니다.

Playwright는 실제 Chromium 브라우저를 띄워 페이지를 완전히 로드한 뒤, iframe 내부의 캡차 엘리먼트를 CSS 셀렉터로 직접 잡아 스크린샷을 찍는 방식으로 이미지를 저장합니다.

### 동작 흐름

```
1. Chromium 브라우저 실행 (headless: false — 실제 창이 뜸)
2. 대법원 나의 사건 검색 페이지 접속
3. 페이지 내 iframe에서 캡차 이미지 엘리먼트 탐색
4. 해당 엘리먼트를 PNG 스크린샷으로 저장
5. 1~2초 랜덤 대기 (서버 부하 방지)
6. 지정 횟수만큼 2~5번 반복
7. 브라우저 종료
```

### 파일명 규칙

출력 파일명은 시작 번호부터 4자리 zero-padding 형식으로 자동 생성됩니다.

```
시작번호 0, 반복 3회 → 0000.png, 0001.png, 0002.png
시작번호 1000, 반복 5회 → 1000.png, 1001.png, 1002.png, 1003.png, 1004.png
```

---

## 설치 방법

### 사전 요구사항

*   **Node.js** 18 이상 설치 필요 ([nodejs.org](https://nodejs.org/))

### 의존성 설치

`scraper/` 폴더로 이동한 뒤 npm으로 패키지를 설치합니다.

```powershell
cd scraper
npm install
```

Playwright를 처음 사용하는 환경이라면, Chromium 브라우저 바이너리도 함께 설치해야 합니다.

```powershell
npx playwright install chromium
```

---

## 사용 방법

```powershell
node app.js [시작번호] [반복횟수]
```

### 파라미터 설명

| 파라미터    | 설명                               | 예시    |
| ----------- | ---------------------------------- | ------- |
| `시작번호`  | 저장할 파일명의 시작 숫자 (0 이상) | `0`     |
| `반복횟수`  | 캡차를 몇 장 수집할지 (1 이상)     | `1000`  |

### 실행 예시

```powershell
# 0000.png 부터 0999.png 까지 1,000장 수집
node app.js 0 1000

# 3000.png 부터 3499.png 까지 500장 수집
node app.js 3000 500
```

실행하면 Chromium 브라우저 창이 열리고, 대법원 사이트에 반복 접속하며 캡차를 한 장씩 수집합니다. 수집된 이미지는 `scraper/downloaded/` 폴더에 저장됩니다.

### 수집한 이미지를 학습에 활용

수집이 끝나면, `downloaded/` 폴더의 이미지를 `training/TrainingData/`로 옮기고 정답 라벨(`answer_key.txt`)을 작성하면 모델 재학습에 바로 투입할 수 있습니다.

---

## 폴더 구조

```
scraper/
├── app.js          # 캡차 수집 메인 스크립트
├── package.json    # Node.js 프로젝트 설정 및 의존성
├── README.md       # 본 문서
└── downloaded/     # (실행 시 자동 생성) 수집된 캡차 이미지 저장 폴더
```

---

## 주의 사항

*   수집 속도를 조절하기 위해 매 요청마다 1~2초의 랜덤 대기 시간이 포함되어 있습니다. 서버에 과부하를 주지 않도록 이 간격을 임의로 줄이지 마세요.
*   `headless: false`로 설정되어 있어 실제 브라우저 창이 뜹니다. 백그라운드 실행이 필요하면 `app.js`에서 `headless: true`로 변경하세요.
