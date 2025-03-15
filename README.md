# 음성 기반 캘린더 비서

이 프로젝트는 음성 인식과 LLM을 활용하여 개인의 캘린더를 관리하는 음성 비서 애플리케이션입니다.

## 주요 기능

1. 음성 인식을 통한 일정 정보 추출
2. 일정 추가, 조회, 수정, 삭제
3. Google 캘린더 연동
4. 다양한 LLM 모델 지원 (Hugging Face, Ollama)

## 시스템 요구사항

- Python 3.8 이상
- PyAudio (음성 녹음용)
- Google API 인증 정보
- (선택) CUDA 지원 GPU (대형 모델 실행 시 권장)

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/voice-calendar.git
cd voice-calendar
```

2. 가상 환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 의존성 설치

```bash
pip install -r requirements.txt
```

4. Google API 인증 설정

- [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
- Google Calendar API 활성화
- OAuth 2.0 클라이언트 ID 생성
- 인증 정보 다운로드 후 `gcloud/credentials.json`에 저장

## 사용 방법

1. 애플리케이션 실행

```bash
python main.py
```

2. 메뉴 선택
   - 음성 녹음 및 일정 추가
   - 오늘 일정 조회
   - 일정 검색
   - 설정 변경

## 설정 파일

`config.yaml` 파일에서 다음 설정을 변경할 수 있습니다:

- 캘린더 설정 (Google 캘린더 인증 정보)
- 음성 인식 설정 (모델, 언어, 녹음 설정)
- LLM 설정 (모델 유형, 모델 이름)

## 프로젝트 구조

```
voice-calendar/
├── main.py                  # 메인 실행 파일
├── config.yaml              # 설정 파일
├── requirements.txt         # 의존성 목록
├── models/                  # 모델 캐시 디렉토리
├── gcloud/                  # Google API 인증 정보
├── src/                     # 소스 코드
│   ├── app.py               # 애플리케이션 클래스
│   ├── calendar/            # 캘린더 모듈
│   │   ├── calendar_interface.py
│   │   └── google_calendar.py
│   ├── speech/              # 음성 인식 모듈
│   │   └── speech_recognizer.py
│   ├── llm/                 # LLM 모듈
│   │   └── llm_processor.py
│   └── utils/               # 유틸리티 모듈
│       └── config.py
└── tests/                   # 테스트 코드
    ├── calendar/
    ├── speech/
    ├── llm/
    └── utils/
```

## 테스트 실행

```bash
python -m unittest discover tests
```

## 라이선스

MIT
