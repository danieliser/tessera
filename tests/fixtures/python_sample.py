"""Analytics module."""

import json
from typing import Optional


class Event:
    """Base event class."""

    def __init__(self, name: str, data: dict):
        self.name = name
        self.data = data

    def serialize(self) -> str:
        return json.dumps({"name": self.name, "data": self.data})


class ClickEvent(Event):
    """Click event with coordinates."""

    def __init__(self, x: int, y: int):
        super().__init__("click", {"x": x, "y": y})
        self.coords = (x, y)

    def distance_from_origin(self) -> float:
        return (self.coords[0] ** 2 + self.coords[1] ** 2) ** 0.5


async def process_event(event: Event) -> Optional[str]:
    """Process an event and return serialized form."""
    result = event.serialize()
    return result


def create_click(x: int, y: int) -> ClickEvent:
    """Factory for click events."""
    click = ClickEvent(x, y)
    click.distance_from_origin()
    return click
