from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

final_state_of_run = client.runs.wait(
    assistant_id="agent",
    thread_id=None,
    input={"messages": [{"role": "user", "content": "how are you?"}]},
    config={"configurable": {"thread_id": "1"}}
)
print(final_state_of_run)

