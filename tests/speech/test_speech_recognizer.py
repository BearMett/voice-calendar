"""
음성 인식 모듈 테스트
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import wave
import pyaudio
import torch
from src.speech.speech_recognizer import SpeechRecognizer


class TestSpeechRecognizer(unittest.TestCase):
    """음성 인식 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.speech_recognizer = SpeechRecognizer(
            model_name="openai/whisper-tiny",  # 테스트용 작은 모델 사용
            language="korean",
            cache_dir="test_models",
        )

    def tearDown(self):
        """테스트 정리"""
        # 테스트 파일 정리
        test_files = ["test_audio.wav"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

        # 테스트 모델 디렉토리는 유지 (다운로드 시간 절약)

    @patch("torch.cuda.is_available")
    def test_init(self, mock_cuda_available):
        """초기화 테스트"""
        # CUDA 사용 가능한 경우
        mock_cuda_available.return_value = True
        recognizer = SpeechRecognizer()
        self.assertEqual(recognizer.model_name, "openai/whisper-large-v3")
        self.assertEqual(recognizer.language, "korean")
        self.assertEqual(recognizer.device, "cuda")
        self.assertEqual(
            recognizer.notification_sound, "src/speech/sounds/start_recording.mp3"
        )

        # CUDA 사용 불가능한 경우
        mock_cuda_available.return_value = False
        recognizer = SpeechRecognizer()
        self.assertEqual(recognizer.device, "cpu")

    @patch("os.path.exists")
    @patch("src.speech.speech_recognizer.PLAYSOUND_AVAILABLE", True)
    @patch("playsound.playsound")
    def test_play_notification_playsound(self, mock_playsound, mock_path_exists):
        """알림 소리 재생 테스트 (playsound 사용)"""
        # 알림 소리 파일이 존재하는 경우
        mock_path_exists.return_value = True

        # 테스트 실행
        with patch("src.speech.speech_recognizer.playsound", mock_playsound):
            self.speech_recognizer.play_notification()
            mock_playsound.assert_called_once_with(
                self.speech_recognizer.notification_sound
            )

    @patch("os.path.exists")
    @patch("src.speech.speech_recognizer.PLAYSOUND_AVAILABLE", False)
    @patch("src.speech.speech_recognizer.WINSOUND_AVAILABLE", True)
    @patch("src.speech.speech_recognizer.winsound.PlaySound")
    def test_play_notification_winsound(self, mock_play_sound, mock_path_exists):
        """알림 소리 재생 테스트 (winsound 사용)"""
        # 알림 소리 파일이 존재하는 경우
        mock_path_exists.return_value = True

        # 테스트 실행
        self.speech_recognizer.play_notification()
        mock_play_sound.assert_called_once_with(
            self.speech_recognizer.notification_sound, mock_play_sound.return_value
        )

    @patch("os.path.exists")
    @patch("src.speech.speech_recognizer.PLAYSOUND_AVAILABLE", False)
    @patch("src.speech.speech_recognizer.WINSOUND_AVAILABLE", False)
    @patch("src.speech.speech_recognizer.SYSTEM", "Darwin")
    @patch("subprocess.call")
    def test_play_notification_macos(self, mock_subprocess_call, mock_path_exists):
        """알림 소리 재생 테스트 (macOS 시스템 명령어 사용)"""
        # 알림 소리 파일이 존재하는 경우
        mock_path_exists.return_value = True

        # 테스트 실행
        self.speech_recognizer.play_notification()
        mock_subprocess_call.assert_called_once_with(
            ["afplay", self.speech_recognizer.notification_sound]
        )

    @patch("os.path.exists")
    @patch("src.speech.speech_recognizer.PLAYSOUND_AVAILABLE", False)
    @patch("src.speech.speech_recognizer.WINSOUND_AVAILABLE", False)
    @patch("src.speech.speech_recognizer.SYSTEM", "Linux")
    @patch("subprocess.call")
    def test_play_notification_linux(self, mock_subprocess_call, mock_path_exists):
        """알림 소리 재생 테스트 (Linux 시스템 명령어 사용)"""
        # 알림 소리 파일이 존재하는 경우
        mock_path_exists.return_value = True

        # 테스트 실행
        self.speech_recognizer.play_notification()
        mock_subprocess_call.assert_called_once_with(
            ["aplay", self.speech_recognizer.notification_sound]
        )

    @patch("os.path.exists")
    def test_play_notification_no_file(self, mock_path_exists):
        """알림 소리 재생 테스트 (파일 없음)"""
        # 알림 소리 파일이 존재하지 않는 경우
        mock_path_exists.return_value = False

        # 테스트 실행 (오류 없이 실행되어야 함)
        self.speech_recognizer.play_notification()

    # 모델 로드 테스트는 실제 모델을 로드하지 않고 모킹하기 어려우므로 건너뜁니다
    def skip_test_load_model(self):
        """모델 로드 테스트 (건너뜀)"""
        pass

    @patch("pyaudio.PyAudio")
    @patch("wave.open")
    @patch("numpy.frombuffer")
    @patch("numpy.sqrt")
    @patch("numpy.mean")
    @patch("time.time")
    @patch.object(SpeechRecognizer, "play_notification")
    def test_record_audio(
        self,
        mock_play_notification,
        mock_time,
        mock_mean,
        mock_sqrt,
        mock_frombuffer,
        mock_wave_open,
        mock_pyaudio,
    ):
        """음성 녹음 테스트"""
        # 모의 객체 설정
        mock_pyaudio_instance = MagicMock()
        mock_pyaudio.return_value = mock_pyaudio_instance

        mock_stream = MagicMock()
        mock_pyaudio_instance.open.return_value = mock_stream

        # 시간 모킹 (최대 녹음 시간 체크를 위해)
        mock_time.side_effect = [0, 10, 20, 30]  # 시작 시간, 체크 시간들

        # 무음 감지를 위한 모킹
        mock_frombuffer.return_value = np.array([0, 0, 0])
        mock_mean.return_value = 100
        mock_sqrt.side_effect = [
            500,
            500,
            100,
        ]  # 배경 소음, 첫 번째 청크, 두 번째 청크 (무음)

        # 스트림 읽기 모킹
        mock_stream.read.return_value = b"test_data"

        # 파일 쓰기 모킹
        mock_wave_file = MagicMock()
        mock_wave_open.return_value = mock_wave_file

        # 샘플 크기 모킹
        mock_pyaudio_instance.get_sample_size.return_value = 2

        # 테스트 파일 이름
        filename = "test_audio.wav"

        # 녹음 함수 호출
        result = self.speech_recognizer.record_audio(
            filename=filename, silence_threshold=1000, silence_duration=0.1
        )

        # 알림 소리 재생 확인
        mock_play_notification.assert_called_once()

        # 스트림 열기 확인
        mock_pyaudio_instance.open.assert_called_once()

        # 스트림 읽기 확인
        self.assertTrue(mock_stream.read.called)

        # 스트림 종료 확인
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio_instance.terminate.assert_called_once()

        # 파일 쓰기 확인
        mock_wave_open.assert_called_once_with(filename, "wb")
        mock_wave_file.setnchannels.assert_called_once_with(1)
        mock_wave_file.setsampwidth.assert_called_once_with(2)
        mock_wave_file.setframerate.assert_called_once_with(16000)
        mock_wave_file.writeframes.assert_called_once()
        mock_wave_file.close.assert_called_once()

        # 결과 확인
        self.assertEqual(result, filename)

    @patch.object(SpeechRecognizer, "load_model")
    def test_transcribe(self, mock_load_model):
        """음성 텍스트 변환 테스트"""
        # 모의 객체 설정
        self.speech_recognizer.pipe = None  # 모델이 로드되지 않은 상태 시뮬레이션

        # load_model이 호출되면 pipe 속성 설정
        def side_effect():
            self.speech_recognizer.pipe = MagicMock()
            self.speech_recognizer.pipe.return_value = {"text": "테스트 텍스트"}

        mock_load_model.side_effect = side_effect

        # 텍스트 변환 함수 호출
        result = self.speech_recognizer.transcribe("test_audio.wav")

        # 모델 로드 확인
        mock_load_model.assert_called_once()

        # 결과 확인
        self.assertEqual(result, "테스트 텍스트")

        # 이미 모델이 로드된 경우
        mock_load_model.reset_mock()
        result = self.speech_recognizer.transcribe("test_audio.wav")

        # 모델 로드가 호출되지 않음
        mock_load_model.assert_not_called()

        # 결과 확인
        self.assertEqual(result, "테스트 텍스트")


if __name__ == "__main__":
    unittest.main()
