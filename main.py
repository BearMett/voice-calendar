"""
음성 기반 캘린더 애플리케이션 메인 실행 파일
"""

from src.app import VoiceCalendarApp


def main():
    """메인 함수"""
    app = VoiceCalendarApp()
    app.run()


if __name__ == "__main__":
    main()
