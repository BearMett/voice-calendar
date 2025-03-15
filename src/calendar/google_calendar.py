"""
Google 캘린더 연동 모듈

이 모듈은 Google 캘린더 API를 사용하여 일정을 관리하는 기능을 제공합니다.
"""

import os
from datetime import datetime
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GoogleCalendarManager:
    """Google 캘린더 API를 사용하여 일정을 관리하는 클래스"""

    def __init__(
        self, credentials_path="./gcloud/credentials.json", token_path="token.json"
    ):
        """
        GoogleCalendarManager 초기화

        Args:
            credentials_path (str): Google API 인증 정보 파일 경로
            token_path (str): 토큰 저장 파일 경로
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = ["https://www.googleapis.com/auth/calendar"]
        self.service = None

    def authenticate(self):
        """Google 캘린더 API 인증 설정"""
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        self.service = build("calendar", "v3", credentials=creds)
        return self.service

    def add_event(self, calendar_info):
        """
        Google 캘린더에 일정 추가

        Args:
            calendar_info (dict): 일정 정보 (날짜, 시간, 제목, 참석자, 장소 등)

        Returns:
            bool: 일정 추가 성공 여부
        """
        try:
            if not self.service:
                self.authenticate()

            # 날짜와 시간 포맷 변환
            date_str = calendar_info.get("date", datetime.now().strftime("%Y-%m-%d"))
            time_str = calendar_info.get("time", "00:00")

            # 시작 시간과 종료 시간 설정 (기본값: 1시간 일정)
            start_datetime = f"{date_str}T{time_str}:00"

            # 종료 시간 계산 (기본 1시간)
            hour, minute = map(int, time_str.split(":"))
            end_hour = hour + 1
            end_minute = minute

            # 시간 조정 (24시간 형식)
            if end_hour >= 24:
                end_hour -= 24

            end_datetime = f"{date_str}T{end_hour:02d}:{end_minute:02d}:00"

            # 참석자 목록 처리
            attendees_list = []
            if (
                calendar_info.get("attendees")
                and calendar_info.get("attendees") != "없음"
            ):
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

            event = (
                self.service.events().insert(calendarId="primary", body=event).execute()
            )
            print(f'일정이 성공적으로 추가되었습니다: {event.get("htmlLink")}')
            return True

        except Exception as e:
            print(f"일정 추가 중 오류 발생: {e}")
            return False

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
        try:
            if not self.service:
                self.authenticate()

            if not start_date:
                start_date = datetime.now().strftime("%Y-%m-%d")

            if not end_date:
                end_date = start_date

            # 시간 형식 변환
            time_min = f"{start_date}T00:00:00Z"
            time_max = f"{end_date}T23:59:59Z"

            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            return events

        except Exception as e:
            print(f"일정 조회 중 오류 발생: {e}")
            return []

    def delete_event(self, event_id):
        """
        일정 삭제

        Args:
            event_id (str): 삭제할 일정의 ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if not self.service:
                self.authenticate()

            self.service.events().delete(
                calendarId="primary", eventId=event_id
            ).execute()
            return True

        except Exception as e:
            print(f"일정 삭제 중 오류 발생: {e}")
            return False

    def update_event(self, event_id, calendar_info):
        """
        일정 수정

        Args:
            event_id (str): 수정할 일정의 ID
            calendar_info (dict): 수정할 일정 정보

        Returns:
            bool: 수정 성공 여부
        """
        try:
            if not self.service:
                self.authenticate()

            # 기존 일정 정보 가져오기
            event = (
                self.service.events()
                .get(calendarId="primary", eventId=event_id)
                .execute()
            )

            # 수정할 정보 업데이트
            if "title" in calendar_info:
                event["summary"] = calendar_info["title"]

            if "location" in calendar_info:
                event["location"] = calendar_info["location"]

            if "date" in calendar_info and "time" in calendar_info:
                date_str = calendar_info["date"]
                time_str = calendar_info["time"]

                # 시작 시간 설정
                start_datetime = f"{date_str}T{time_str}:00"
                event["start"]["dateTime"] = start_datetime

                # 종료 시간 계산 (기본 1시간)
                hour, minute = map(int, time_str.split(":"))
                end_hour = hour + 1
                end_minute = minute

                # 시간 조정 (24시간 형식)
                if end_hour >= 24:
                    end_hour -= 24

                end_datetime = f"{date_str}T{end_hour:02d}:{end_minute:02d}:00"
                event["end"]["dateTime"] = end_datetime

            # 참석자 목록 처리
            if "attendees" in calendar_info and calendar_info["attendees"] != "없음":
                attendees_list = []
                for attendee in calendar_info["attendees"].split(","):
                    attendees_list.append({"email": attendee.strip()})
                event["attendees"] = attendees_list

            # 일정 업데이트
            updated_event = (
                self.service.events()
                .update(calendarId="primary", eventId=event_id, body=event)
                .execute()
            )

            return True

        except Exception as e:
            print(f"일정 수정 중 오류 발생: {e}")
            return False

    def search_events(self, query, max_results=10):
        """
        일정 검색

        Args:
            query (str): 검색어
            max_results (int): 최대 결과 수

        Returns:
            list: 검색된 일정 목록
        """
        try:
            if not self.service:
                self.authenticate()

            # 현재 날짜부터 향후 일정 검색
            time_min = datetime.now().isoformat() + "Z"

            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                    q=query,
                )
                .execute()
            )

            events = events_result.get("items", [])
            return events

        except Exception as e:
            print(f"일정 검색 중 오류 발생: {e}")
            return []
