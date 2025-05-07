from __future__ import annotations
from redel.tools import ToolBase
from kani import AIFunction

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import datetime


class GoogleCalendarTool(ToolBase):
    def __init__(self, app, kani, service_account_info: dict, calendar_id="primary"):
        super().__init__(app, kani)
        self.calendar_id = calendar_id
        self.creds = Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        self.service = build("calendar", "v3", credentials=self.creds)

    @AIFunction(name="create_google_meet", desc="Create a Google Calendar event with a Meet link")
    async def create_google_meet(
        self,
        summary: str,
        description: str,
        start_time: str,  # Format: 2025-05-07T14:00:00
        end_time: str,
        timezone: str = "America/Los_Angeles",
    ) -> str:
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_time, "timeZone": timezone},
            "end": {"dateTime": end_time, "timeZone": timezone},
            "conferenceData": {
                "createRequest": {
                    "requestId": f"meet-{datetime.datetime.now().timestamp()}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }

        created_event = (
            self.service.events()
            .insert(calendarId=self.calendar_id, body=event, conferenceDataVersion=1)
            .execute()
        )

        return created_event["hangoutLink"]
