"""
음성 기반 캘린더 애플리케이션 메인 실행 파일

이 애플리케이션은 음성 인식과 텍스트 입력을 통해 캘린더에 일정을 추가하고 관리할 수 있습니다.
음성 명령과 텍스트 명령 모두 지원합니다.
"""

from src.app import VoiceCalendarApp


def main():
    """메인 함수"""
    app = VoiceCalendarApp()
    app.run()


if __name__ == "__main__":
    main()
