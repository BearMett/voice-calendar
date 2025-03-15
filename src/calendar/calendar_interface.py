"""
캘린더 인터페이스 모듈

이 모듈은 캘린더 서비스에 대한 통합 인터페이스를 제공합니다.
현재는 Google 캘린더만 지원하지만, 향후 다른 캘린더 서비스도 추가할 수 있습니다.
"""

from src.calendar.google_calendar import GoogleCalendarManager


class CalendarInterface:
    """캘린더 서비스에 대한 통합 인터페이스 클래스"""

    def __init__(self, calendar_type="google", **kwargs):
        """
        CalendarInterface 초기화

        Args:
            calendar_type (str): 사용할 캘린더 서비스 유형 (현재는 'google'만 지원)
            **kwargs: 캘린더 서비스별 추가 인자
        """
        self.calendar_type = calendar_type
        self.calendar_manager = None

        if calendar_type == "google":
            credentials_path = kwargs.get("credentials_path", "gcloud/credentials.json")
            token_path = kwargs.get("token_path", "token.json")
            self.calendar_manager = GoogleCalendarManager(
                credentials_path=credentials_path, token_path=token_path
            )
        else:
            raise ValueError(f"지원하지 않는 캘린더 유형: {calendar_type}")

    def authenticate(self):
        """캘린더 서비스 인증"""
        return self.calendar_manager.authenticate()

    def add_event(self, calendar_info):
        """
        일정 추가

        Args:
            calendar_info (dict): 일정 정보 (날짜, 시간, 제목, 참석자, 장소 등)

        Returns:
            bool: 일정 추가 성공 여부
        """
        return self.calendar_manager.add_event(calendar_info)

    def get_events(self, start_date=None, end_date=None, max_results=10):
        """
        특정 기간의 일정 조회

        Args:
            start_date (str): 시작 날짜 (YYYY-MM-DD 형식)
            end_date (str): 종료 날짜 (YYYY-MM-DD 형식)
            max_results (int): 최대 결과 수

        Returns:
            list: 일정 목록
        """
        return self.calendar_manager.get_events(start_date, end_date, max_results)

    def delete_event(self, event_id):
        """
        일정 삭제

        Args:
            event_id (str): 삭제할 일정의 ID

        Returns:
            bool: 삭제 성공 여부
        """
        return self.calendar_manager.delete_event(event_id)

    def update_event(self, event_id, calendar_info):
        """
        일정 수정

        Args:
            event_id (str): 수정할 일정의 ID
            calendar_info (dict): 수정할 일정 정보

        Returns:
            bool: 수정 성공 여부
        """
        return self.calendar_manager.update_event(event_id, calendar_info)

    def search_events(self, query, max_results=10):
        """
        일정 검색

        Args:
            query (str): 검색어
            max_results (int): 최대 결과 수

        Returns:
            list: 검색된 일정 목록
        """
        return self.calendar_manager.search_events(query, max_results)

    def format_event_summary(self, event):
        """
        일정 요약 정보 포맷팅

        Args:
            event (dict): 일정 정보

        Returns:
            str: 포맷팅된 일정 요약 정보
        """
        summary = event.get("summary", "제목 없음")

        # 시작 시간 포맷팅
        start = event.get("start", {})
        if "dateTime" in start:
            start_time = start["dateTime"].split("T")
            start_date = start_time[0]
            start_time = start_time[1].split("+")[0][:5]  # HH:MM 형식으로 변환
            start_str = f"{start_date} {start_time}"
        else:
            start_str = start.get("date", "날짜 없음")

        # 종료 시간 포맷팅
        end = event.get("end", {})
        if "dateTime" in end:
            end_time = end["dateTime"].split("T")
            end_date = end_time[0]
            end_time = end_time[1].split("+")[0][:5]  # HH:MM 형식으로 변환
            end_str = f"{end_date} {end_time}"
        else:
            end_str = end.get("date", "날짜 없음")

        location = event.get("location", "장소 없음")

        return f"제목: {summary}\n시작: {start_str}\n종료: {end_str}\n장소: {location}"

    def format_events_list(self, events):
        """
        일정 목록 포맷팅

        Args:
            events (list): 일정 목록

        Returns:
            str: 포맷팅된 일정 목록
        """
        if not events:
            return "일정이 없습니다."

        result = []
        for i, event in enumerate(events, 1):
            event_summary = self.format_event_summary(event)
            result.append(f"[{i}] {event_summary}\n")

        return "\n".join(result)
