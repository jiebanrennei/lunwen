import requests
import json
import hashlib
import time
import uuid

SECRET = "961b98075f8d0b0644285"  # 和环境变量 TUITUI_SECRET 一致

url = "http://127.0.0.1:8088/api/ai-assistant/bot/webhook/"
payload = {
    "cid": "7652886633113456",
    "uid": "7652669334456778",
    "user_account": "zhangsan",
    "user_name": "张三",
    "timestamp": "1591337058",
    "event": "single_chat",
    "data": {
        "msgid": "123****",
        "msg_type": "text",
        "text": "这条是消息正文"
    }
}

body = json.dumps(payload, ensure_ascii=False)
timestamp = str(int(time.time()))
nonce = uuid.uuid4().hex

# 计算签名：sha1(secret + timestamp + nonce + body)
raw = (SECRET + timestamp + nonce + body).encode('utf-8')
checksum = hashlib.sha1(raw).hexdigest()

headers = {
    "Content-Type": "application/json",
    "X-Tuitui-Robot-Timestamp": timestamp,
    "X-Tuitui-Robot-Nonce": nonce,
    "X-Tuitui-Robot-Checksum": checksum,
}

response = requests.post(url, data=body.encode('utf-8'), headers=headers, timeout=30)
print(response.status_code)
print(response.text)
