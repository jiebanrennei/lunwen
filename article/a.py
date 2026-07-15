import base64
import requests

api = 'http://10.237.197.215:7777/download'
url = "https://www.quark.cn/s?q=宝可梦"
data = {
    "url": url
}


def save_base64_img(b64_str, save_path="screenshot.png"):
    """把 base64 截图字符串解码并保存为图片文件, 返回保存路径。"""
    if not b64_str or len(b64_str) < 100:
        return None
    # 兼容 data URI 前缀 (data:image/png;base64,....)
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    with open(save_path, "wb") as f:
        f.write(img_bytes)
    return save_path


response = requests.post(url=api, data=data, timeout=30)
res = response.json()

# 截图之外的字段照常打印, 避免超长 base64 刷屏
b64_img = res.get("screenshot", "")
preview = {k: v for k, v in res.items() if k != "screenshot"}
preview["screenshot_len"] = len(b64_img)
print(preview)

path = save_base64_img(b64_img)
if path:
    print(f"截图已保存: {path} (base64 长度 {len(b64_img)})")
else:
    print("无有效截图数据")