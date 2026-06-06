import os
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
body = json.dumps({
    "model": "qwen2.5:0.5b",
    "prompt": "What is a neural network in one sentence?",
    "stream": False,
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["response"])
