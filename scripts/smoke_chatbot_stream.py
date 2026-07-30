import requests
import json

# create thread
res = requests.post("http://localhost:2024/threads", json={})
thread_id = res.json()["thread_id"]

# stream
payload = {
    "assistant_id": "agent",
    "input": {"messages": [{"type": "human", "content": "hi"}]},
    "stream_mode": ["messages"]
}

with requests.post(f"http://localhost:2024/threads/{thread_id}/runs/stream", json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode('utf-8'))
