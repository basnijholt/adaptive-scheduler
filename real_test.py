from dotenv import load_dotenv

load_dotenv()
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from adaptive_scheduler._server_support.llm_manager import LLMManager

# Create minimal setup
db_manager = MagicMock()

# Test the interrupt logic without needing API keys
print("🧪 Testing interrupt logic without external APIs...")

with patch("adaptive_scheduler._server_support.llm_manager.ChatGoogleGenerativeAI") as mock_llm:
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance

    # Create LLMManager with Google (but mocked)
    llm_manager = LLMManager(
        db_manager=db_manager,
        model_provider="google",
        model_name="gemini-2.5-flash",
        yolo=False,  # Enable approval flow
    )

    # Mock the LLM to return a message with tool calls for write_file
    mock_ai_response = AIMessage(
        content="I'll write the file for you.",
        tool_calls=[
            {
                "id": "call_123",
                "name": "write_file",
                "args": {"file_path": "test_output.py", "text": "print('hello world')"},
            },
        ],
    )

    # Mock the ainvoke method to return our mock response
    mock_llm_instance.bind_tools.return_value.ainvoke = AsyncMock(return_value=mock_ai_response)


async def test_interrupt():
    thread_id = "test_thread"

    # Check what tools are available
    tools = llm_manager.toolkit.get_tools()
    print(f"📋 Available tools: {[tool.name for tool in tools]}")

    # Be very explicit about using the write_file tool
    message = """I need you to use the write_file tool to create a file.

    Use write_file with these parameters:
    - file_path: test_output.py
    - text: print("hello world")

    Please call the write_file tool now."""

    print(f"🤖 Sending: {message}")

    # Let's inspect what happens during the chat
    print("🔍 Calling llm_manager.chat...")

    # Test the chat method which should detect the interrupt
    print("📤 Testing chat method (should trigger interrupt)")
    result = await llm_manager.chat(message, thread_id=thread_id)

    if result.interrupted:
        print(f"🚫 INTERRUPT: {result.interrupt_message}")

        # Auto-approve for testing (interrupt expects just the decision string)
        approval_data = "approved"
        print(f"🤖 Auto-approving with: {approval_data}")

        print(f"📤 Resuming with: {approval_data}")
        resume_result = await llm_manager.resume_chat(approval_data, thread_id=thread_id)
        print(f"✅ Resume response: '{resume_result.content}'")
    else:
        print(f"❌ UNEXPECTED: Chat returned without interrupt: '{result.content}'")


# Run it
if __name__ == "__main__":
    asyncio.run(test_interrupt())
