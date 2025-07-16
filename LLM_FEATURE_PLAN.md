# LLM Feature Implementation Plan

This document outlines the plan for integrating an LLM-powered chat feature into the Adaptive Scheduler. The goal is to create a system that can automatically diagnose failed jobs and allow users to interact with an LLM to understand job statuses and results.

## Todo

- [x] **1. Create `LLMManager`**
    - [x] Create `adaptive_scheduler/_server_support/llm_manager.py`.
    - [x] Implement the `LLMManager` class, inheriting from `BaseManager`.
    - [ ] **Refactor for Async:** Convert `diagnose_failed_job` and `chat` to `async` methods to avoid blocking I/O.
    - [x] Implement `diagnose_failed_job(job_id)`:
        - [x] Fetch the log file for the failed job.
        - [x] Implement a (simulated) LLM call to analyze the log and identify the root cause of the error.
        - [x] Store the diagnosis in a cache.
    - [x] Implement `chat(message)`:
        - [x] Maintain a conversation history.
        - [x] Implement a (simulated) LLM call to generate a response based on the message and history.

- [x] **2. Integrate with `JobManager`**
    - [x] Modify `JobManager` to detect failed jobs.
    - [ ] **Refactor for Async:** Update `JobManager` to `await` the async `diagnose_failed_job` method.

- [x] **3. Orchestrate with `RunManager`**
    - [x] Update `RunManager` to initialize and manage `LLMManager`.
    - [x] Make the `LLMManager` instance available to the UI.

- [x] **4. Enhance the UI**
    - [x] Add a "Chat with LLM" button to the `info` widget in `widgets.py`.
    - [x] Create a `chat_widget` function to build the chat interface.
    - [ ] **Refactor for Async:** Update the chat widget to handle async interactions with the `LLMManager`.
    - [x] When opening the chat for a failed job, display the pre-computed diagnosis.
    - [x] Handle interactive chat, displaying the conversation history.

- [ ] **5. Finalize and Test**
    - [ ] **Increase Test Coverage:**
        - [ ] Add tests for async file I/O in `LLMManager`.
        - [ ] Add tests for the async integration between `JobManager` and `LLMManager`.
        - [ ] Add tests for the async UI interactions in the `chat_widget`.
    - [x] Write unit tests for the new components.
    - [x] Perform end-to-end testing in a Jupyter notebook.
