import time
import requests

api = "http://spider.crawler.agent.qihoo.net:7777/download"
data = {
    "url": 'https://www.weather.com.cn/weather/101010100.shtml',
    "markdown_out": 0,
    "force_js": 1,
    "need_image": 0,
    "screenshot": 0,
    "wait_time": 15,
    "rolling": 0,
    "dp_crawler": 1,
    "content_out": 1,
    'page_date_deadpage': 0,
    'video_api_info': 0,
    'need_check': 0,
}
# 恢复 data 表单提交，去掉 json=
response = requests.post(api, data=data, timeout=30)
res = response.json()
print("返回结果：", res)

# 判断采集是否成功
if res.get("code") != 0 or not res.get("html"):
    print("采集失败：", res.get("info"))
else:
    import base64, zlib
    b64_str = res["html"]
    pad = 4 - len(b64_str) % 4
    if pad != 4:
        b64_str += "=" * pad
    byte_data = base64.b64decode(b64_str)
    html = zlib.decompress(byte_data).decode("utf-8", errors="ignore")
    print(html)