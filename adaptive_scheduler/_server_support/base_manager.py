from __future__ import annotations

import abc
import asyncio
from collections.abc import Coroutine


class _BaseManager(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.ioloop: asyncio.events.AbstractEventLoop | None = None
        self._coro: Coroutine | None = None
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        if self.is_started:
            raise Exception(f"{self.__class__} is already started!")
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
            return self.task.cancel()
        return None

    def _setup(self) -> None:
        """Is run in the beginning of `self.start`."""

    @abc.abstractmethod
    async def _manage(self) -> None:  # pragma: no cover
        pass
