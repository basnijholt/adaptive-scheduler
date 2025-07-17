#!/usr/bin/env python3
"""Test LLMManager with real LLM (no mocking)."""

from dotenv import load_dotenv

load_dotenv()
import asyncio
from unittest.mock import MagicMock

from adaptive_scheduler._server_support.llm_manager import LLMManager


async def test_real_llm():
    """Test with a real LLM to see the actual interrupt flow using ChatResult."""
    # Create minimal setup
    db_manager = MagicMock()

    # Create LLMManager with Google (real API)
    llm_manager = LLMManager(
        db_manager=db_manager,
        model_provider="google",
        model_name="gemini-2.5-flash",
        yolo=False,  # Enable approval flow
    )

    thread_id = "test_thread"

    # Check what tools are available
    tools = llm_manager.toolkit.get_tools()
    print(f"📋 Available tools: {[tool.name for tool in tools]}")

    # Ask the LLM to write a file (should trigger interrupt)
    message = """Please create a Python file called 'test_real.py' that prints 'Hello from real LLM!'

    Use the write_file tool to create this file."""

    print(f"🤖 Sending: {message}")

    try:
        print("🔍 Calling real LLM...")
        result = await llm_manager.chat(message, thread_id=thread_id)

        if result.interrupted:
            print(f"🚫 INTERRUPT: {result.interrupt_message}")

            # Auto-approve for testing
            approval_data = "approved"
            print(f"🤖 Auto-approving: {approval_data}")

            print(f"📤 Resuming with: {approval_data}")
            try:
                resume_result = await llm_manager.resume_chat(approval_data, thread_id=thread_id)
                print(f"✅ Resume response: '{resume_result.content}'")

                # Check if file was created
                try:
                    with open("test_real.py") as f:
                        content = f.read()
                    print(f"📄 File created successfully with content: {content!r}")
                except FileNotFoundError:
                    print("❌ File was not created")

            except Exception as resume_error:
                print(f"❌ Resume error: {resume_error}")
        else:
            print(f"❌ UNEXPECTED: Chat returned without interrupt: '{result.content}'")

    except Exception as e:
        print(f"❌ Error: {e}")


# Run it
if __name__ == "__main__":
    asyncio.run(test_real_llm())
