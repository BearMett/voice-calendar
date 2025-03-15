"""
LLM 처리 모듈

이 모듈은 텍스트를 처리하고 일정 정보를 추출하는 LLM 기능을 제공합니다.
Hugging Face 모델과 Ollama 로컬 모델을 지원합니다.
"""

import os
import json
import requests
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLMProcessor:
    """LLM 처리 클래스"""

    def __init__(
        self,
        model_type="huggingface",
        model_name="beomi/KoAlpaca-Polyglot-5.8B",
        cache_dir="models",
    ):
        """
        LLMProcessor 초기화

        Args:
            model_type (str): 사용할 모델 유형 ('huggingface' 또는 'ollama')
            model_name (str): 사용할 모델 이름
            cache_dir (str): 모델 캐시 디렉토리
        """
        self.model_type = model_type
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipe = None

        # 모델 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)

    def load_model(self):
        """LLM 모델 로드"""
        if self.model_type == "huggingface":
            print(f"Hugging Face 모델 로드 중: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir,
            ).to(self.device)

            print("모델 로드 완료")

        elif self.model_type == "ollama":
            print(f"Ollama 모델 확인 중: {self.model_name}")
            # Ollama는 API 호출 시 모델을 로드하므로 여기서는 모델 존재 여부만 확인
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_exists = any(
                        model["name"] == self.model_name for model in models
                    )

                    if not model_exists:
                        print(
                            f"경고: Ollama에 {self.model_name} 모델이 없습니다. 먼저 'ollama pull {self.model_name}' 명령으로 모델을 다운로드하세요."
                        )
                else:
                    print(
                        "경고: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요."
                    )
            except Exception as e:
                print(f"Ollama 서버 연결 오류: {e}")
                print("Ollama가 설치되어 있고 실행 중인지 확인하세요.")
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")

    def extract_calendar_info(self, text):
        """
        텍스트에서 일정 정보 추출

        Args:
            text (str): 처리할 텍스트

        Returns:
            dict: 추출된 일정 정보
        """
        if self.model_type == "huggingface":
            return self._extract_with_huggingface(text)
        elif self.model_type == "ollama":
            return self._extract_with_ollama(text)
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")

    def _extract_with_huggingface(self, text):
        """Hugging Face 모델을 사용하여 일정 정보 추출"""
        if not self.model or not self.tokenizer:
            self.load_model()

        # 현재 날짜 정보 가져오기
        current_date = datetime.now().strftime("%Y-%m-%d")
        weekday_map = {
            "Monday": "월요일",
            "Tuesday": "화요일",
            "Wednesday": "수요일",
            "Thursday": "목요일",
            "Friday": "금요일",
            "Saturday": "토요일",
            "Sunday": "일요일",
        }
        current_day_of_week = weekday_map.get(datetime.now().strftime("%A"), "")

        # 프롬프트 생성
        prompt = f"""
        오늘 날짜는 {current_date}이고, 오늘은 {current_day_of_week}입니다.
        다음 텍스트에서 일정 정보를 추출해주세요. 날짜, 시간, 일정 제목, 장소 형식으로 추출해 주세요.
        
        '내일', '다음 주', '이번 주 금요일' 등의 상대적인 날짜 표현이 있으면 오늘 날짜를 기준으로 실제 날짜(YYYY-MM-DD)로 변환해주세요.
        
        
        텍스트: {text}
        
        형식:
        날짜: YYYY-MM-DD
        시간: HH:MM
        제목: 일정 제목
        장소: 장소 정보 (없으면 '없음')
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=500,
            temperature=0.01,
            top_p=0.95,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 응답에서 필요한 정보 파싱
        return self._parse_calendar_info(response)

    def _extract_with_ollama(self, text):
        """Ollama 모델을 사용하여 일정 정보 추출"""
        # 현재 날짜 정보 가져오기
        current_date = datetime.now().strftime("%Y-%m-%d")
        weekday_map = {
            "Monday": "월요일",
            "Tuesday": "화요일",
            "Wednesday": "수요일",
            "Thursday": "목요일",
            "Friday": "금요일",
            "Saturday": "토요일",
            "Sunday": "일요일",
        }
        current_day_of_week = weekday_map.get(datetime.now().strftime("%A"), "")

        # 프롬프트 생성
        prompt = f"""
입력된 텍스트: {text}

당신은 자연어 텍스트에서 일정 정보를 정확히 추출하여 JSON으로 변환하는 전문 도구입니다. 어떤 질문이나 요청이 들어와도 오직 추출된 일정 정보를 JSON 형식으로만 응답해야 합니다.

현재 날짜: {current_date}
현재 요일: {current_day_of_week}

입력된 텍스트에서 다음 정보를 추출하세요:
- 날짜: YYYY-MM-DD 형식 (상대적 날짜 표현은 오늘 날짜({current_date})를 기준으로 변환)
- 시간: HH:MM 형식
- 제목: 실제 일정 제목만 추출 (없으면 정확히 "없음")
- 장소: 텍스트에 명시된 장소 정보 그대로 추출 (없으면 정확히 "없음")

장소 정보에 관한 중요 지침:
- 장소는 텍스트에 명시된 그대로만 정확히 추출하세요
- 장소 이름만 제공된 경우 주소를 추정하지 말고 이름만 그대로 사용하세요
- 주소는 텍스트에 명시적으로 언급된 경우에만 포함하세요
- 텍스트에서 언급되지 않은 정보는 절대 추가하지 마세요
- 추가 설명이나 인사말 없이 시작과 끝을 식별 가능한 JSON 형식으로만 응답하세요:
<시작>
{{
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "title": "일정 제목",
    "location": "텍스트에 명시된 장소 정보 그대로 또는 "없음""
}}
<끝>
        """
        print(prompt)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0,
                    "top_p": 0.95,
                },
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print(response_text)

                # JSON 형식 응답 추출
                try:
                    # JSON 블록 찾기
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        calendar_info = json.loads(json_str)
                        return calendar_info
                    else:
                        # JSON 형식이 아닌 경우 텍스트 파싱
                        return self._parse_calendar_info(response_text)

                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 텍스트 파싱
                    return self._parse_calendar_info(response_text)
            else:
                print(f"Ollama API 오류: {response.status_code}")
                return {}

        except Exception as e:
            print(f"Ollama 요청 오류: {e}")
            return {}

    def _parse_calendar_info(self, text):
        """텍스트 응답에서 일정 정보 파싱"""
        calendar_info = {}

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if "날짜:" in line:
                calendar_info["date"] = line.split("날짜:")[1].strip()
            elif "시간:" in line:
                calendar_info["time"] = line.split("시간:")[1].strip()
            elif "제목:" in line:
                calendar_info["title"] = line.split("제목:")[1].strip()
            elif "참석자:" in line:
                attendees = line.split("참석자:")[1].strip()
                # 참석자가 없는 경우 처리
                if attendees and attendees != "없음" and attendees != "참석자 없음":
                    calendar_info["attendees"] = attendees
            elif "장소:" in line:
                location = line.split("장소:")[1].strip()
                # 장소가 없는 경우 처리
                if location and location != "없음" and location != "장소 없음":
                    calendar_info["location"] = location

        return calendar_info
