import os
from datetime import datetime
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pyaudio
import wave
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from google.oauth2 import service_account


# 1. 음성 녹음 기능
def record_audio(seconds=5, filename="recorded_audio.wav"):
    """마이크로 음성을 녹음하는 함수"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print(f"{seconds}초 동안 녹음합니다...")
    frames = []

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 완료!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return filename


# 2. 음성을 텍스트로 변환 (Hugging Face ASR 모델 사용)
def speech_to_text(audio_file):
    """Hugging Face의 Whisper 모델을 사용하여 음성을 텍스트로 변환"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3").to(
        device
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        device=device,
    )

    # 한국어 음성 인식 설정
    result = pipe(audio_file, generate_kwargs={"language": "korean"})

    return result["text"]


# 3. 텍스트에서 일정 정보 추출 (Hugging Face LLM 모델 사용)
def extract_calendar_info(text):
    """LLM을 사용하여 텍스트에서 일정 정보 추출"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 한국어 지원 모델 로드
    model_name = "beomi/KoAlpaca-Polyglot-5.8B"  # 한국어 지원되는 모델 선택
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 프롬프트 생성
    prompt = f"""
    다음 텍스트에서 일정 정보를 추출해주세요. 날짜, 시간, 일정 제목, 참석자, 장소 형식으로 추출해 주세요.
    
    텍스트: {text}
    
    형식:
    날짜: YYYY-MM-DD
    시간: HH:MM
    제목: 일정 제목
    참석자: 참석자 목록 (없으면 '없음')
    장소: 장소 정보 (없으면 '없음')
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, max_length=500, temperature=0.1, top_p=0.95, do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 응답에서 필요한 정보 파싱
    calendar_info = {}

    lines = response.split("\n")
    for line in lines:
        if "날짜:" in line:
            calendar_info["date"] = line.split("날짜:")[1].strip()
        elif "시간:" in line:
            calendar_info["time"] = line.split("시간:")[1].strip()
        elif "제목:" in line:
            calendar_info["title"] = line.split("제목:")[1].strip()
        elif "참석자:" in line:
            calendar_info["attendees"] = line.split("참석자:")[1].strip()
        elif "장소:" in line:
            calendar_info["location"] = line.split("장소:")[1].strip()

    return calendar_info


# 4. Google 캘린더 연동
def setup_google_calendar():
    """Google 캘린더 API 인증 설정"""
    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "gcloud/credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    return service


def add_event_to_calendar(service, calendar_info):
    """Google 캘린더에 일정 추가"""
    try:
        # 날짜와 시간 포맷 변환
        date_str = calendar_info.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = calendar_info.get("time", "00:00")

        # 시작 시간과 종료 시간 설정 (기본값: 1시간 일정)
        start_datetime = f"{date_str}T{time_str}:00"
        end_datetime = f"{date_str}T{time_str.split(':')[0]}:{int(time_str.split(':')[1]) + 60:02d}:00"

        # 참석자 목록 처리
        attendees_list = []
        if calendar_info.get("attendees") and calendar_info.get("attendees") != "없음":
            for attendee in calendar_info.get("attendees").split(","):
                attendees_list.append({"email": attendee.strip()})

        # 일정 생성
        event = {
            "summary": calendar_info.get("title", "새 일정"),
            "location": calendar_info.get("location", ""),
            "description": "음성 인식을 통해 생성된 일정",
            "start": {
                "dateTime": start_datetime,
                "timeZone": "Asia/Seoul",
            },
            "end": {
                "dateTime": end_datetime,
                "timeZone": "Asia/Seoul",
            },
            "attendees": attendees_list,
            "reminders": {
                "useDefault": True,
            },
        }

        event = service.events().insert(calendarId="primary", body=event).execute()
        print(f'일정이 성공적으로 추가되었습니다: {event.get("htmlLink")}')
        return True

    except Exception as e:
        print(f"일정 추가 중 오류 발생: {e}")
        return False


# 5. 메인 애플리케이션 실행 함수
def main():
    """음성 기반 캘린더 애플리케이션 메인 함수"""
    print("음성 기반 캘린더 애플리케이션을 시작합니다.")

    while True:
        print("\n1. 음성 녹음 및 일정 추가")
        print("2. 종료")
        choice = input("선택하세요: ")

        if choice == "1":
            # 1. 음성 녹음
            audio_file = record_audio(seconds=10)

            # 2. 음성을 텍스트로 변환
            text = speech_to_text(audio_file)
            print(f"인식된 텍스트: {text}")

            # 3. 텍스트에서 일정 정보 추출
            calendar_info = extract_calendar_info(text)
            print("\n추출된 일정 정보:")
            for key, value in calendar_info.items():
                print(f"{key}: {value}")

            # 사용자 확인
            confirm = input("\n이 정보로 일정을 추가하시겠습니까? (y/n): ")
            if confirm.lower() == "y":
                # 4. Google 캘린더 설정
                service = setup_google_calendar()

                # 5. 캘린더에 일정 추가
                success = add_event_to_calendar(service, calendar_info)
                if success:
                    print("일정이 성공적으로 추가되었습니다.")
                else:
                    print("일정 추가에 실패했습니다.")

        elif choice == "2":
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")


if __name__ == "__main__":
    main()
