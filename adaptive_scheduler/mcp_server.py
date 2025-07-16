"""A server for handling MCP requests."""

from __future__ import annotations

import asyncio
from typing import Any

from adaptive_scheduler._server_support.llm_manager import LLMManager


class MCPServer:
    """A server for handling MCP requests."""

    def __init__(self, llm_manager: LLMManager) -> None:
        """Initialize the MCP server."""
        self.llm_manager = llm_manager

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handles a request and returns a response."""
        if request["method"] == "diagnose_failed_job":
            job_id = request["params"]["job_id"]
            diagnosis = await self.llm_manager.diagnose_failed_job(job_id)
            return {"result": diagnosis}
        if request["method"] == "chat":
            message = request["params"]["message"]
            response = await self.llm_manager.chat(message)
            return {"result": response}
        return {"error": "Unknown method"}


async def main() -> None:
    """The main entry point for the server."""
    llm_manager = LLMManager()
    server = MCPServer(llm_manager)
    # This is a placeholder for the actual server implementation.
    # In a real application, this would be a long-running process
    # that listens for requests on a socket.
    request = {
        "method": "chat",
        "params": {"message": "Hello, world!"},
    }
    response = await server.handle_request(request)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
