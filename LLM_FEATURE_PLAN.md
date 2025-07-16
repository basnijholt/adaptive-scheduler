# LLM Feature Implementation Plan

This document outlines the plan for integrating an LLM-powered chat feature into the Adaptive Scheduler. The goal is to create a system that can automatically diagnose failed jobs and allow users to interact with an LLM to understand job statuses and results.

## Todo

- [x] **1. Create `LLMManager`**
    - [x] Create `adaptive_scheduler/_server_support/llm_manager.py`.
    - [x] Implement the `LLMManager` class, inheriting from `BaseManager`.
    - [x] **Refactor for Async:** Convert `diagnose_failed_job` and `chat` to `async` methods to avoid blocking I/O.
    - [x] Implement `diagnose_failed_job(job_id)`:
        - [x] Fetch the log file for the failed job.
        - [x] Implement a (simulated) LLM call to analyze the log and identify the root cause of the error.
        - [x] Store the diagnosis in a cache.
    - [x] Implement `chat(message)`:
        - [x] Maintain a conversation history.
        - [x] Implement a (simulated) LLM call to generate a response based on the message and history.

- [x] **2. Integrate with `JobManager`**
    - [x] Modify `JobManager` to detect failed jobs.
    - [x] **Refactor for Async:** Update `JobManager` to `await` the async `diagnose_failed_job` method.

- [x] **3. Orchestrate with `RunManager`**
    - [x] Update `RunManager` to initialize and manage `LLMManager`.
    - [x] Make the `LLMManager` instance available to the UI.

- [x] **4. Enhance the UI**
    - [x] Add a "Chat with LLM" button to the `info` widget in `widgets.py`.
    - [x] Create a `chat_widget` function to build the chat interface.
    - [x] **Refactor for Async:** Update the chat widget to handle async interactions with the `LLMManager`.
    - [x] When opening the chat for a failed job, display the pre-computed diagnosis.
    - [x] Handle interactive chat, displaying the conversation history.

- [x] **5. Code Cleanup and Refinement**
    - [x] Review existing implementation for unused code, debugging leftovers, and clarity.
    - [x] Refactor `llm_manager.py` and `widgets.py` for better readability and maintainability.
    - [x] Ensure all existing functionality is robust before adding new features.

- [x] **6. Add Tools to the AI**
    - [x] Integrate `langchain`'s file system tools into `LLMManager`.
    - [x] Allow the AI to read and write files.
    - [x] Fine-tune the tool integration to ensure reliability and security.
    - [ ] Read the documentation of `langchain` tools to ensure we are using the most appropriate and up to date tools for our needs.

- [x] **7. Implement "Approve" and "YOLO" Modes**
    - [x] Create a mechanism for the user to approve changes suggested by the AI (e.g., typing "approve").
    - [x] Add a "YOLO" mode to allow the AI to make changes without user confirmation.
    - [x] Refine the user interaction flow for clarity and ease of use.

- [ ] **8. Migrate to `langgraph`**
    - [x] Replace `AgentExecutor` with `create_react_agent` from `langgraph`.
    - [x] Update `LLMManager` to use the new `langgraph` agent.
    - [x] Refactor tests to align with the new `langgraph` implementation.
    - [ ] Fix remaining test failures.

- [ ] **8. Improve Chat Widget UI**
    - [ ] Use a JavaScript library (e.g., `marked.js` or similar) to render Markdown in the chat widget.
    - [ ] Add syntax highlighting for code blocks.
    - [ ] Polish the visual appearance of the chat bubbles and overall layout.

- [ ] **9. Finalize and Test**
    - [x] **Increase Test Coverage:**
        - [x] Add tests for async file I/O in `LLMManager`.
        - [x] Add tests for the async integration between `JobManager` and `LLMManager`.
        - [x] Add tests for the async UI interactions in the `chat_widget`.
        - [x] Add tests for the AI tools and approve/YOLO modes.
    - [ ] Write unit tests for the new components.
    - [ ] Perform end-to-end testing in a Jupyter notebook.
