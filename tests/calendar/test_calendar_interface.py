"""
캘린더 인터페이스 테스트 모듈
"""

import unittest
from unittest.mock import patch, MagicMock
from src.calendar.calendar_interface import CalendarInterface
from src.calendar.google_calendar import GoogleCalendarManager


class TestCalendarInterface(unittest.TestCase):
    """캘린더 인터페이스 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        # GoogleCalendarManager.authenticate 메서드를 모킹하여 파일 시스템 접근 방지
        self.patcher = patch.object(GoogleCalendarManager, "authenticate")
        self.mock_authenticate = self.patcher.start()
        self.mock_authenticate.return_value = True

        # 다른 메서드들도 모킹
        self.patcher_add = patch.object(GoogleCalendarManager, "add_event")
        self.mock_add_event = self.patcher_add.start()
        self.mock_add_event.return_value = True

        self.patcher_get = patch.object(GoogleCalendarManager, "get_events")
        self.mock_get_events = self.patcher_get.start()
        self.mock_get_events.return_value = []

        self.patcher_delete = patch.object(GoogleCalendarManager, "delete_event")
        self.mock_delete_event = self.patcher_delete.start()
        self.mock_delete_event.return_value = True

        self.patcher_update = patch.object(GoogleCalendarManager, "update_event")
        self.mock_update_event = self.patcher_update.start()
        self.mock_update_event.return_value = True

        self.patcher_search = patch.object(GoogleCalendarManager, "search_events")
        self.mock_search_events = self.patcher_search.start()
        self.mock_search_events.return_value = []

        # CalendarInterface 인스턴스 생성
        self.calendar_interface = CalendarInterface()

    def tearDown(self):
        """테스트 종료 시 패치 중지"""
        self.patcher.stop()
        self.patcher_add.stop()
        self.patcher_get.stop()
        self.patcher_delete.stop()
        self.patcher_update.stop()
        self.patcher_search.stop()

    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.calendar_interface.calendar_type, "google")
        self.assertIsNotNone(self.calendar_interface.calendar_manager)

    def test_authenticate(self):
        """인증 테스트"""
        result = self.calendar_interface.authenticate()
        self.mock_authenticate.assert_called_once()
        self.assertTrue(result)

    def test_add_event(self):
        """일정 추가 테스트"""
        calendar_info = {
            "date": "2023-03-15",
            "time": "14:00",
            "title": "테스트 일정",
            "attendees": "test@example.com",
            "location": "테스트 장소",
        }

        result = self.calendar_interface.add_event(calendar_info)
        self.mock_add_event.assert_called_once_with(calendar_info)
        self.assertTrue(result)

    def test_get_events(self):
        """일정 조회 테스트"""
        result = self.calendar_interface.get_events("2023-03-15", "2023-03-16", 5)
        self.mock_get_events.assert_called_once_with("2023-03-15", "2023-03-16", 5)
        self.assertEqual(result, [])

    def test_delete_event(self):
        """일정 삭제 테스트"""
        result = self.calendar_interface.delete_event("event123")
        self.mock_delete_event.assert_called_once_with("event123")
        self.assertTrue(result)

    def test_update_event(self):
        """일정 수정 테스트"""
        event_id = "event123"
        calendar_info = {"title": "수정된 일정", "location": "수정된 장소"}

        result = self.calendar_interface.update_event(event_id, calendar_info)
        self.mock_update_event.assert_called_once_with(event_id, calendar_info)
        self.assertTrue(result)

    def test_search_events(self):
        """일정 검색 테스트"""
        result = self.calendar_interface.search_events("테스트", 5)
        self.mock_search_events.assert_called_once_with("테스트", 5)
        self.assertEqual(result, [])

    def test_format_event_summary(self):
        """일정 요약 포맷팅 테스트"""
        event = {
            "summary": "테스트 일정",
            "start": {"dateTime": "2023-03-15T14:00:00+09:00"},
            "end": {"dateTime": "2023-03-15T15:00:00+09:00"},
            "location": "테스트 장소",
        }

        expected_summary = "제목: 테스트 일정\n시작: 2023-03-15 14:00\n종료: 2023-03-15 15:00\n장소: 테스트 장소"
        actual_summary = self.calendar_interface.format_event_summary(event)
        self.assertEqual(actual_summary, expected_summary)

    def test_format_events_list(self):
        """일정 목록 포맷팅 테스트"""
        # 빈 목록 테스트
        self.assertEqual(
            self.calendar_interface.format_events_list([]), "일정이 없습니다."
        )

        # 일정 목록 테스트
        events = [
            {
                "summary": "테스트 일정 1",
                "start": {"dateTime": "2023-03-15T14:00:00+09:00"},
                "end": {"dateTime": "2023-03-15T15:00:00+09:00"},
                "location": "테스트 장소 1",
            },
            {
                "summary": "테스트 일정 2",
                "start": {"dateTime": "2023-03-16T10:00:00+09:00"},
                "end": {"dateTime": "2023-03-16T11:00:00+09:00"},
                "location": "테스트 장소 2",
            },
        ]

        # format_event_summary 메서드를 모킹하여 테스트
        with patch.object(
            self.calendar_interface, "format_event_summary"
        ) as mock_format:
            mock_format.side_effect = [
                "제목: 테스트 일정 1\n시작: 2023-03-15 14:00\n종료: 2023-03-15 15:00\n장소: 테스트 장소 1",
                "제목: 테스트 일정 2\n시작: 2023-03-16 10:00\n종료: 2023-03-16 11:00\n장소: 테스트 장소 2",
            ]

            actual_list = self.calendar_interface.format_events_list(events)

            # 실제 결과 확인
            expected_list = (
                "[1] 제목: 테스트 일정 1\n시작: 2023-03-15 14:00\n종료: 2023-03-15 15:00\n장소: 테스트 장소 1\n\n"
                "[2] 제목: 테스트 일정 2\n시작: 2023-03-16 10:00\n종료: 2023-03-16 11:00\n장소: 테스트 장소 2\n"
            )

            self.assertEqual(actual_list, expected_list)
            self.assertEqual(mock_format.call_count, 2)


if __name__ == "__main__":
    unittest.main()
