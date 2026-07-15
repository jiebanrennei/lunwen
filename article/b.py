import time
import base64
import requests

api = "http://spider.crawler.agent.qihoo.net:7777/download"
target_url = 'https://www.quark.cn/s?q=宝可梦'
payload = {
    "url": target_url,
    "markdown_out": 0,
    "force_js": 1,
    "need_image": 0,
    "screenshot": 1,
    "wait_time": 8,    # 适当缩短页面等待，减轻耗时
    "rolling": 0,
    "dp_crawler": 1,
    "content_out": 0,
    'page_date_deadpage': 0,
    'video_api_info': 0,
    'need_check': 0,
}

def safe_parse_json(resp):
    text = resp.text.strip()
    if not text:
        return None, "接口返回空内容"
    try:
        return resp.json(), None
    except Exception as e:
        return None, f"JSON解析失败,原始返回:{repr(text)}"

def save_base64_img(b64_str, save_path="screenshot.jpg"):
    if b64_str.startswith("data:"):
        b64_data = b64_str.split(",", 1)[1]
    else:
        b64_data = b64_str
    img_bytes = base64.b64decode(b64_data)
    with open(save_path, "wb") as f:
        f.write(img_bytes)
    return save_path

def crawl_with_retry(max_retry=2):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    # 爬虫内部wait_time=8，外层请求超时给到28秒预留缓冲
    timeout = 28
    total_start = time.perf_counter()  # 全部请求(含重试/等待)总耗时起点
    for attempt in range(1, max_retry + 1):
        print(f"【第{attempt}次请求爬虫】{time.strftime('%Y-%m-%d %H:%M:%S')}")
        req_start = time.perf_counter()  # 本次请求耗时起点
        try:
            response = requests.post(url=api, data=payload, headers=headers, timeout=timeout)
            req_elapsed = time.perf_counter() - req_start
            print(f"第{attempt}次请求耗时: {req_elapsed:.2f}s")
            res_data, err = safe_parse_json(response)
            if err:
                print(f"返回解析异常：{err}")
                time.sleep(2)
                continue
            total_elapsed = time.perf_counter() - total_start
            print(f"爬取成功，总耗时(含重试/等待): {total_elapsed:.2f}s")
            return res_data, None
        except requests.exceptions.ReadTimeout:
            print(f"第{attempt}次读取超时({time.perf_counter() - req_start:.2f}s)，等待2s重试")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"第{attempt}次无法连接爬虫服务({time.perf_counter() - req_start:.2f}s)")
            time.sleep(2)
        except Exception as e:
            print(f"爬虫未知异常({time.perf_counter() - req_start:.2f}s): {str(e)}")
            time.sleep(2)
    total_elapsed = time.perf_counter() - total_start
    print(f"全部请求失败，总耗时(含重试/等待): {total_elapsed:.2f}s")
    return None, "多次请求爬虫均超时/失败"

if __name__ == "__main__":
    prog_start = time.perf_counter()  # 端到端总耗时起点
    res_data, err_msg = crawl_with_retry(max_retry=2)
    if err_msg or not res_data:
        print("爬虫调用最终失败：", err_msg)
    else:
        print("爬虫返回正常")
        html = res_data.get("html", "")
        b64_img = res_data.get("screenshot")
        if b64_img and len(b64_img) > 100:
            path = save_base64_img(b64_img)
            print(f"截图保存至：{path}，base64长度：{len(b64_img)}")
        else:
            print("无有效截图数据")
    print(f"端到端总耗时(爬取+解析+存图): {time.perf_counter() - prog_start:.2f}s")