from typing import Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"


# Step 1: Define data models for routing and responses
class CalendarRequestType(BaseModel):
    """Router LLM Call: Determines the type of calendar request

    Args:
        BaseModel
    """

    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar requst being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new event

    Args:
        BaseModel
    """

    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: str = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    """Details for changing an existing event

    Args:
        BaseModel
    """

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event

    Args:
        BaseModel
    """

    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(description="New participants to add")
    participants_to_remove: list[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format

    Args:
        BaseModel
    """

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if possible")


# Step 2: Define the routing and processing functions
def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of the calendar request.

    Args:
        user_input (str): Natural language calendar request from the user.

    Returns:
        CalendarRequestType
    """
    logger.info("Routing Calendar Request")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event or modify an existing one.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=CalendarRequestType,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
    )
    return result


def handle_new_event(description: str) -> CalendarResponse:
    """Process a new event request

    Args:
        description (str): The description of the new event.

    Returns:
        CalendarResponse
    """
    logger.info("Processing new event request")

    # Get event details
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details for creating a new calendar event.",
            },
            {"role": "user", "content": description},
        ],
        response_format=NewEventDetails,
    )
    details = completion.choices[0].message.parsed
    logger.info(f"New event: {details.model_dump_json(indent = 2)}")

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


def handle_modify_event(description: str) -> CalendarResponse:
    """Processes an event modification request

    Args:
        description (str): The description of the event to be modified.

    Returns:
        CalendarResponse
    """
    logger.info("Processing event modification request")

    # Get modification details
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details for modifying an existing calendar event",
            },
            {"role": "user", "content": description},
        ],
        response_format=ModifyEventDetails,
    )
    details = completion.choices[0].message.parsed
    logger.info(f"Modified event: {details.model_dump_json(indent = 2)}")

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}' with the requested change.",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )
