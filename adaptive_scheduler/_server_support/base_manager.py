from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine


class ManagerAlreadyStartedError(Exception):
    """Raised when a manager is already started."""


class BaseManager(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.ioloop: asyncio.events.AbstractEventLoop | None = None
        self._coro: Coroutine | None = None
        self.task: asyncio.Task | None = None

    def start(self) -> BaseManager:
        if self.is_started:
            msg = f"{self.__class__} is already started!"
            raise ManagerAlreadyStartedError(msg)
        self._setup()
        self.ioloop = asyncio.get_event_loop()
        self._coro = self._manage()
        self.task = self.ioloop.create_task(self._coro)
        return self

    @property
    def is_started(self) -> bool:
        return self.task is not None

    def cancel(self) -> bool | None:
        if self.is_started:
            assert self.task is not None  # for mypy
            return self.task.cancel()
        return None

    def _setup(self) -> None:  # noqa: B027
        """Is run in the beginning of `self.start`."""

    @abc.abstractmethod
    async def _manage(self) -> None:  # pragma: no cover
        pass
