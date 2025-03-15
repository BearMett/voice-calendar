"""
음성 기반 캘린더 애플리케이션

이 모듈은 음성 인식, LLM, 캘린더 모듈을 통합하여 음성 기반 캘린더 애플리케이션을 제공합니다.
"""

import os
from src.utils.config import Config
from src.calendar.calendar_interface import CalendarInterface
from src.speech.speech_recognizer import SpeechRecognizer
from src.llm.llm_processor import LLMProcessor


class VoiceCalendarApp:
    """음성 기반 캘린더 애플리케이션 클래스"""

    def __init__(self, config_path="config.yaml"):
        """
        VoiceCalendarApp 초기화

        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = Config(config_path)

        # 캘린더 모듈 초기화
        calendar_config = self.config.get("calendar")
        self.calendar = CalendarInterface(
            calendar_type=calendar_config.get("type", "google"),
            credentials_path=calendar_config.get(
                "credentials_path", "gcloud/credentials.json"
            ),
            token_path=calendar_config.get("token_path", "token.json"),
        )

        # 음성 인식 모듈 초기화
        speech_config = self.config.get("speech")
        self.speech_recognizer = SpeechRecognizer(
            model_name=speech_config.get("model_name", "openai/whisper-large-v3"),
            language=speech_config.get("language", "korean"),
            cache_dir=speech_config.get("cache_dir", "models"),
        )

        # LLM 모듈 초기화
        llm_config = self.config.get("llm")
        self.llm_processor = LLMProcessor(
            model_type=llm_config.get("model_type", "huggingface"),
            model_name=llm_config.get("model_name", "beomi/KoAlpaca-Polyglot-5.8B"),
            cache_dir=llm_config.get("cache_dir", "models"),
        )

    def run(self):
        """애플리케이션 실행"""
        print("음성 기반 캘린더 애플리케이션을 시작합니다.")

        while True:
            print("\n1. 음성 녹음 및 일정 추가")
            print("2. 오늘 일정 조회")
            print("3. 일정 검색")
            print("4. 설정 변경")
            print("5. 종료")
            choice = input("선택하세요: ")

            if choice == "1":
                self.record_and_add_event()
            elif choice == "2":
                self.show_today_events()
            elif choice == "3":
                self.search_events()
            elif choice == "4":
                self.change_settings()
            elif choice == "5":
                print("프로그램을 종료합니다.")
                break
            else:
                print("잘못된 선택입니다. 다시 시도해주세요.")

    def record_and_add_event(self):
        """음성 녹음 및 일정 추가"""
        try:
            # 음성 녹음
            speech_config = self.config.get("speech")
            audio_file = self.speech_recognizer.record_audio(
                silence_threshold=speech_config.get("silence_threshold", 1000),
                silence_duration=speech_config.get("silence_duration", 2.0),
                max_duration=speech_config.get("max_duration", 60),
            )

            # 음성을 텍스트로 변환
            text = self.speech_recognizer.transcribe(audio_file)
            print(f"\n인식된 텍스트: {text}")

            # 텍스트에서 일정 정보 추출
            calendar_info = self.llm_processor.extract_calendar_info(text)
            print("\n추출된 일정 정보:")
            for key, value in calendar_info.items():
                print(f"{key}: {value}")

            # 사용자 확인
            confirm = input("\n이 정보로 일정을 추가하시겠습니까? (y/n): ")
            if confirm.lower() == "y":
                # 캘린더에 일정 추가
                success = self.calendar.add_event(calendar_info)
                if success:
                    print("일정이 성공적으로 추가되었습니다.")
                else:
                    print("일정 추가에 실패했습니다.")

        except Exception as e:
            print(f"오류 발생: {e}")

    def show_today_events(self):
        """오늘 일정 조회"""
        try:
            # 오늘 일정 조회
            events = self.calendar.get_events()

            # 일정 목록 출력
            events_text = self.calendar.format_events_list(events)
            print("\n오늘 일정:")
            print(events_text)

        except Exception as e:
            print(f"오류 발생: {e}")

    def search_events(self):
        """일정 검색"""
        try:
            # 검색어 입력
            query = input("검색어를 입력하세요: ")

            # 일정 검색
            events = self.calendar.search_events(query)

            # 일정 목록 출력
            events_text = self.calendar.format_events_list(events)
            print(f"\n'{query}' 검색 결과:")
            print(events_text)

            # 일정 수정 또는 삭제
            if events:
                action = input(
                    "\n일정을 수정하거나 삭제하시겠습니까? (1: 수정, 2: 삭제, 3: 취소): "
                )

                if action in ["1", "2"]:
                    event_idx = int(input("일정 번호를 입력하세요: ")) - 1

                    if 0 <= event_idx < len(events):
                        event = events[event_idx]
                        event_id = event["id"]

                        if action == "1":  # 수정
                            print(
                                "\n수정할 정보를 입력하세요 (변경하지 않으려면 빈칸으로 두세요):"
                            )
                            title = input(f"제목 ({event.get('summary', '')}): ")
                            date = input(f"날짜 (YYYY-MM-DD): ")
                            time = input(f"시간 (HH:MM): ")
                            location = input(f"장소 ({event.get('location', '')}): ")

                            # 수정할 정보 구성
                            calendar_info = {}
                            if title:
                                calendar_info["title"] = title
                            if date:
                                calendar_info["date"] = date
                            if time:
                                calendar_info["time"] = time
                            if location:
                                calendar_info["location"] = location

                            # 일정 수정
                            success = self.calendar.update_event(
                                event_id, calendar_info
                            )
                            if success:
                                print("일정이 성공적으로 수정되었습니다.")
                            else:
                                print("일정 수정에 실패했습니다.")

                        elif action == "2":  # 삭제
                            confirm = input(
                                f"'{event.get('summary', '')}' 일정을 삭제하시겠습니까? (y/n): "
                            )
                            if confirm.lower() == "y":
                                success = self.calendar.delete_event(event_id)
                                if success:
                                    print("일정이 성공적으로 삭제되었습니다.")
                                else:
                                    print("일정 삭제에 실패했습니다.")
                    else:
                        print("잘못된 일정 번호입니다.")

        except Exception as e:
            print(f"오류 발생: {e}")

    def change_settings(self):
        """설정 변경"""
        print("\n설정 변경:")
        print("1. 음성 인식 설정")
        print("2. LLM 설정")
        print("3. 취소")
        choice = input("선택하세요: ")

        if choice == "1":
            # 음성 인식 설정 변경
            speech_config = self.config.get("speech")

            print("\n현재 음성 인식 설정:")
            for key, value in speech_config.items():
                print(f"{key}: {value}")

            print("\n변경할 설정을 입력하세요 (변경하지 않으려면 빈칸으로 두세요):")
            model_name = input(f"모델 이름 ({speech_config.get('model_name')}): ")
            language = input(f"언어 ({speech_config.get('language')}): ")
            silence_threshold = input(
                f"무음 임계값 ({speech_config.get('silence_threshold')}): "
            )
            silence_duration = input(
                f"무음 지속 시간 ({speech_config.get('silence_duration')}): "
            )
            max_duration = input(
                f"최대 녹음 시간 ({speech_config.get('max_duration')}): "
            )

            # 설정 업데이트
            new_config = {}
            if model_name:
                new_config["model_name"] = model_name
            if language:
                new_config["language"] = language
            if silence_threshold:
                new_config["silence_threshold"] = int(silence_threshold)
            if silence_duration:
                new_config["silence_duration"] = float(silence_duration)
            if max_duration:
                new_config["max_duration"] = int(max_duration)

            if new_config:
                self.config.update("speech", new_config)
                print("음성 인식 설정이 업데이트되었습니다.")

                # 음성 인식 모듈 재초기화
                speech_config = self.config.get("speech")
                self.speech_recognizer = SpeechRecognizer(
                    model_name=speech_config.get("model_name"),
                    language=speech_config.get("language"),
                    cache_dir=speech_config.get("cache_dir"),
                )

        elif choice == "2":
            # LLM 설정 변경
            llm_config = self.config.get("llm")

            print("\n현재 LLM 설정:")
            for key, value in llm_config.items():
                print(f"{key}: {value}")

            print("\n변경할 설정을 입력하세요 (변경하지 않으려면 빈칸으로 두세요):")
            model_type = input(f"모델 유형 ({llm_config.get('model_type')}): ")
            model_name = input(f"모델 이름 ({llm_config.get('model_name')}): ")

            # 설정 업데이트
            new_config = {}
            if model_type:
                new_config["model_type"] = model_type
            if model_name:
                new_config["model_name"] = model_name

            if new_config:
                self.config.update("llm", new_config)
                print("LLM 설정이 업데이트되었습니다.")

                # LLM 모듈 재초기화
                llm_config = self.config.get("llm")
                self.llm_processor = LLMProcessor(
                    model_type=llm_config.get("model_type"),
                    model_name=llm_config.get("model_name"),
                    cache_dir=llm_config.get("cache_dir"),
                )
