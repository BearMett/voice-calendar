"""
음성 인식 모듈

이 모듈은 마이크로부터 음성을 녹음하고, 음성을 텍스트로 변환하는 기능을 제공합니다.
"""

import os
import time
import wave
import platform
import subprocess
import numpy as np
import pyaudio
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# 플랫폼 확인
SYSTEM = platform.system()

# playsound 모듈 가져오기 시도
try:
    from playsound import playsound

    PLAYSOUND_AVAILABLE = True
except ImportError as e:
    print(f"playsound 모듈을 가져오는 중 오류 발생: {e}")
    PLAYSOUND_AVAILABLE = False

# Windows에서 winsound 모듈 가져오기 시도
if SYSTEM == "Windows":
    try:
        import winsound

        WINSOUND_AVAILABLE = True
    except ImportError:
        WINSOUND_AVAILABLE = False
else:
    WINSOUND_AVAILABLE = False


class SpeechRecognizer:
    """음성 인식 클래스"""

    def __init__(
        self,
        model_name="openai/whisper-large-v3",
        language="korean",
        cache_dir="models",
        notification_sound="src/speech/sounds/start_recording.mp3",
    ):
        """
        SpeechRecognizer 초기화

        Args:
            model_name (str): 사용할 음성 인식 모델 이름
            language (str): 인식할 언어
            cache_dir (str): 모델 캐시 디렉토리
            notification_sound (str): 녹음 시작 알림 소리 파일 경로
        """
        self.model_name = model_name
        self.language = language
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.pipe = None
        self.notification_sound = notification_sound

        # 모델 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)

        # 알림 소리 디렉토리 생성
        os.makedirs(os.path.dirname(notification_sound), exist_ok=True)

    def play_notification(self):
        """녹음 시작 알림 소리 재생"""
        if not os.path.exists(self.notification_sound):
            print("알림 소리 파일이 없습니다. 알림 소리 없이 녹음을 시작합니다.")
            return

        # 방법 1: playsound 라이브러리 사용
        if PLAYSOUND_AVAILABLE:
            try:
                playsound(self.notification_sound)
                return
            except Exception as e:
                print(f"playsound를 사용한 알림 소리 재생 중 오류 발생: {e}")

        # 방법 2: Windows에서 winsound 사용
        if WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(self.notification_sound, winsound.SND_FILENAME)
                return
            except Exception as e:
                print(f"winsound를 사용한 알림 소리 재생 중 오류 발생: {e}")

        # 방법 3: 시스템 명령어 사용
        try:
            if SYSTEM == "Darwin":  # macOS
                subprocess.call(["afplay", self.notification_sound])
                return
            elif SYSTEM == "Linux":
                subprocess.call(["aplay", self.notification_sound])
                return
        except Exception as e:
            print(f"시스템 명령어를 사용한 알림 소리 재생 중 오류 발생: {e}")

        print(
            "알림 소리를 재생할 수 있는 방법이 없습니다. 알림 소리 없이 녹음을 시작합니다."
        )

    def load_model(self):
        """음성 인식 모델 로드"""
        print(f"음성 인식 모델 로드 중: {self.model_name}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir,
        ).to(self.device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            device=self.device,
        )

        print("모델 로드 완료")

    def record_audio(
        self,
        filename="recorded_audio.wav",
        silence_threshold=1000,
        silence_duration=2.0,
        max_duration=60,
    ):
        """
        마이크로부터 음성 녹음 (자동 종료 기능 포함)

        Args:
            filename (str): 녹음 파일 저장 경로
            silence_threshold (int): 무음 감지 임계값
            silence_duration (float): 무음 감지 지속 시간 (초)
            max_duration (int): 최대 녹음 시간 (초)

        Returns:
            str: 녹음 파일 경로
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("녹음을 시작합니다. 말씀해 주세요. (무음이 감지되면 자동으로 종료됩니다)")

        # 녹음 시작 알림 소리 재생
        self.play_notification()

        frames = []
        silent_chunks = 0
        silent_threshold = int(silence_duration * RATE / CHUNK)
        start_time = time.time()

        # 처음 0.5초 동안의 배경 소음 레벨 측정
        background_frames = []
        for _ in range(int(0.5 * RATE / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            background_frames.append(data)

        background_data = b"".join(background_frames)
        background_samples = np.frombuffer(background_data, dtype=np.int16)
        background_rms = np.sqrt(np.mean(background_samples**2))

        # 배경 소음 레벨에 따라 임계값 조정
        adjusted_threshold = max(silence_threshold, background_rms * 2)

        print(f"배경 소음 레벨: {background_rms}, 조정된 임계값: {adjusted_threshold}")

        # 실제 녹음 시작
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # 현재 청크의 RMS 값 계산
            current_samples = np.frombuffer(data, dtype=np.int16)
            current_rms = np.sqrt(np.mean(current_samples**2))

            # 무음 감지
            if current_rms < adjusted_threshold:
                silent_chunks += 1
                if silent_chunks >= silent_threshold:
                    print("무음이 감지되어 녹음을 종료합니다.")
                    break
            else:
                silent_chunks = 0

            # 최대 녹음 시간 체크
            if time.time() - start_time > max_duration:
                print(
                    f"최대 녹음 시간 ({max_duration}초)에 도달하여 녹음을 종료합니다."
                )
                break

        print("녹음 완료!")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # 녹음 파일 저장
        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        return filename

    def transcribe(self, audio_file):
        """
        음성 파일을 텍스트로 변환

        Args:
            audio_file (str): 음성 파일 경로

        Returns:
            str: 변환된 텍스트
        """
        if not self.pipe:
            self.load_model()

        print(f"음성을 텍스트로 변환 중...")

        # 한국어 음성 인식 설정
        result = self.pipe(
            audio_file,
            generate_kwargs={
                "language": self.language,
                "max_new_tokens": 128,
                "forced_decoder_ids": None,  # language 설정이 우선되도록 None으로 설정
            },
        )

        return result["text"]
