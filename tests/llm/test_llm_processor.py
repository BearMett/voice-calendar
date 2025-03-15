"""
LLM 처리 모듈 테스트
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import torch
import requests
from src.llm.llm_processor import LLMProcessor


class TestLLMProcessor(unittest.TestCase):
    """LLM 처리 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.llm_processor = LLMProcessor(
            model_type="huggingface",
            model_name="beomi/KoAlpaca-Polyglot-5.8B",
            cache_dir="test_models",
        )

    @patch("torch.cuda.is_available")
    def test_init(self, mock_cuda_available):
        """초기화 테스트"""
        # CUDA 사용 가능한 경우
        mock_cuda_available.return_value = True
        processor = LLMProcessor()
        self.assertEqual(processor.model_type, "huggingface")
        self.assertEqual(processor.model_name, "beomi/KoAlpaca-Polyglot-5.8B")
        self.assertEqual(processor.device, "cuda")

        # CUDA 사용 불가능한 경우
        mock_cuda_available.return_value = False
        processor = LLMProcessor()
        self.assertEqual(processor.device, "cpu")

        # Ollama 모델 유형 테스트
        processor = LLMProcessor(model_type="ollama", model_name="llama2")
        self.assertEqual(processor.model_type, "ollama")
        self.assertEqual(processor.model_name, "llama2")

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_load_model_huggingface(self, mock_model, mock_tokenizer):
        """Hugging Face 모델 로드 테스트"""
        # 모의 객체 설정
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        # 모델 로드 호출
        self.llm_processor.load_model()

        # 함수 호출 확인
        mock_tokenizer.assert_called_once_with(
            self.llm_processor.model_name, cache_dir=self.llm_processor.cache_dir
        )

        mock_model.assert_called_once()

        # 객체 속성 설정 확인
        self.assertIsNotNone(self.llm_processor.tokenizer)
        self.assertIsNotNone(self.llm_processor.model)

    @patch("requests.get")
    def test_load_model_ollama(self, mock_get):
        """Ollama 모델 로드 테스트"""
        # Ollama 모델 설정
        self.llm_processor.model_type = "ollama"
        self.llm_processor.model_name = "llama2"

        # 모의 응답 설정
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2"}]}
        mock_get.return_value = mock_response

        # 모델 로드 호출
        self.llm_processor.load_model()

        # 함수 호출 확인
        mock_get.assert_called_once_with("http://localhost:11434/api/tags")

        # 모델이 없는 경우 테스트
        mock_response.json.return_value = {"models": [{"name": "other-model"}]}
        self.llm_processor.load_model()  # 경고 메시지만 출력하고 예외는 발생하지 않음

        # 서버 연결 오류 테스트
        mock_get.side_effect = requests.exceptions.ConnectionError("연결 오류")
        self.llm_processor.load_model()  # 경고 메시지만 출력하고 예외는 발생하지 않음

    def test_extract_calendar_info(self):
        """일정 정보 추출 테스트"""
        # 지원하지 않는 모델 유형 테스트
        self.llm_processor.model_type = "unsupported"
        with self.assertRaises(ValueError):
            self.llm_processor.extract_calendar_info("테스트 텍스트")

        # Hugging Face 모델 테스트
        self.llm_processor.model_type = "huggingface"
        with patch.object(
            self.llm_processor, "_extract_with_huggingface"
        ) as mock_extract:
            mock_extract.return_value = {"date": "2023-03-15", "time": "14:00"}
            result = self.llm_processor.extract_calendar_info("테스트 텍스트")
            mock_extract.assert_called_once_with("테스트 텍스트")
            self.assertEqual(result, {"date": "2023-03-15", "time": "14:00"})

        # Ollama 모델 테스트
        self.llm_processor.model_type = "ollama"
        with patch.object(self.llm_processor, "_extract_with_ollama") as mock_extract:
            mock_extract.return_value = {"date": "2023-03-15", "time": "14:00"}
            result = self.llm_processor.extract_calendar_info("테스트 텍스트")
            mock_extract.assert_called_once_with("테스트 텍스트")
            self.assertEqual(result, {"date": "2023-03-15", "time": "14:00"})

    @patch.object(LLMProcessor, "load_model")
    def test_extract_with_huggingface(self, mock_load_model):
        """Hugging Face 모델을 사용한 일정 정보 추출 테스트"""
        # 모의 객체 설정
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        self.llm_processor.tokenizer = mock_tokenizer
        self.llm_processor.model = mock_model

        # 모의 응답 설정
        mock_tokenizer.return_value = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_tokenizer.decode.return_value = """
        날짜: 2023-03-15
        시간: 14:00
        제목: 테스트 일정
        참석자: test@example.com
        장소: 테스트 장소
        """

        # 일정 정보 추출 호출
        result = self.llm_processor._extract_with_huggingface(
            "내일 오후 2시에 테스트 미팅이 있습니다."
        )

        # 결과 확인
        self.assertEqual(result["date"], "2023-03-15")
        self.assertEqual(result["time"], "14:00")
        self.assertEqual(result["title"], "테스트 일정")
        self.assertEqual(result["attendees"], "test@example.com")
        self.assertEqual(result["location"], "테스트 장소")

        # 모델이 로드되지 않은 경우 테스트
        self.llm_processor.tokenizer = None
        self.llm_processor.model = None

        # load_model이 호출되면 tokenizer와 model 속성 설정
        def side_effect():
            self.llm_processor.tokenizer = mock_tokenizer
            self.llm_processor.model = mock_model

        mock_load_model.side_effect = side_effect

        # 일정 정보 추출 호출
        result = self.llm_processor._extract_with_huggingface(
            "내일 오후 2시에 테스트 미팅이 있습니다."
        )

        # 함수 호출 확인
        mock_load_model.assert_called_once()

    @patch("requests.post")
    def test_extract_with_ollama(self, mock_post):
        """Ollama 모델을 사용한 일정 정보 추출 테스트"""
        # 모의 응답 설정 (JSON 형식)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": """
            {
                "date": "2023-03-15",
                "time": "14:00",
                "title": "테스트 일정",
                "attendees": "test@example.com",
                "location": "테스트 장소"
            }
            """
        }
        mock_post.return_value = mock_response

        # 일정 정보 추출 호출
        result = self.llm_processor._extract_with_ollama(
            "내일 오후 2시에 테스트 미팅이 있습니다."
        )

        # 함수 호출 확인
        mock_post.assert_called_once()

        # 결과 확인
        self.assertEqual(result["date"], "2023-03-15")
        self.assertEqual(result["time"], "14:00")
        self.assertEqual(result["title"], "테스트 일정")
        self.assertEqual(result["attendees"], "test@example.com")
        self.assertEqual(result["location"], "테스트 장소")

        # JSON 형식이 아닌 응답 테스트
        mock_response.json.return_value = {
            "response": """
            날짜: 2023-03-16
            시간: 15:00
            제목: 다른 테스트 일정
            참석자: other@example.com
            장소: 다른 테스트 장소
            """
        }

        # 일정 정보 추출 호출
        result = self.llm_processor._extract_with_ollama(
            "내일 오후 3시에 다른 테스트 미팅이 있습니다."
        )

        # 결과 확인
        self.assertEqual(result["date"], "2023-03-16")
        self.assertEqual(result["time"], "15:00")
        self.assertEqual(result["title"], "다른 테스트 일정")
        self.assertEqual(result["attendees"], "other@example.com")
        self.assertEqual(result["location"], "다른 테스트 장소")

        # 서버 오류 테스트
        mock_response.status_code = 500
        result = self.llm_processor._extract_with_ollama("테스트 텍스트")
        self.assertEqual(result, {})

        # 연결 오류 테스트
        mock_post.side_effect = requests.exceptions.ConnectionError("연결 오류")
        result = self.llm_processor._extract_with_ollama("테스트 텍스트")
        self.assertEqual(result, {})

    def test_parse_calendar_info(self):
        """일정 정보 파싱 테스트"""
        text = """
        날짜: 2023-03-15
        시간: 14:00
        제목: 테스트 일정
        참석자: test@example.com
        장소: 테스트 장소
        """

        result = self.llm_processor._parse_calendar_info(text)

        self.assertEqual(result["date"], "2023-03-15")
        self.assertEqual(result["time"], "14:00")
        self.assertEqual(result["title"], "테스트 일정")
        self.assertEqual(result["attendees"], "test@example.com")
        self.assertEqual(result["location"], "테스트 장소")

        # 일부 정보만 있는 경우 테스트
        text = """
        날짜: 2023-03-15
        제목: 테스트 일정
        """

        result = self.llm_processor._parse_calendar_info(text)

        self.assertEqual(result["date"], "2023-03-15")
        self.assertEqual(result["title"], "테스트 일정")
        self.assertNotIn("time", result)
        self.assertNotIn("attendees", result)
        self.assertNotIn("location", result)


if __name__ == "__main__":
    unittest.main()
