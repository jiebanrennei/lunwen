#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import sys
import urllib.parse
import urllib.request


UPLOAD_URL = "https://alarm.im.qihoo.net/media/upload"


def encode_multipart_formdata(field_name, file_path):
    boundary = "----python-upload-media-boundary"
    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_data = f.read()

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        file_data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    return body, headers


def upload_media(appid, secret, file_path, media_type="image", timeout=60):
    appid = appid.strip()
    secret = secret.strip()
    media_type = media_type.strip()

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_size = os.path.getsize(file_path)
    max_size = 100 * 1024 * 1024
    if file_size > max_size:
        raise ValueError(f"文件超过 100MB 限制: {file_size} bytes")

    query = urllib.parse.urlencode({
        "appid": appid,
        "secret": secret,
        "type": media_type,
    })
    url = f"{UPLOAD_URL}?{query}"
    body, headers = encode_multipart_formdata("media", file_path)

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def main():
    parser = argparse.ArgumentParser(description="上传文件到 alarm.im.qihoo.net，返回 media_id")
    parser.add_argument("--appid", required=True, help="应用 appid")
    parser.add_argument("--secret", required=True, help="应用 secret")
    parser.add_argument("--file", required=True, help="要上传的文件路径")
    parser.add_argument("--type", default="image", choices=["image", "file"], help="文件类型，默认 image")
    parser.add_argument("--timeout", type=int, default=60, help="请求超时时间，默认 60 秒")
    args = parser.parse_args()

    try:
        result = upload_media(args.appid, args.secret, args.file, args.type, args.timeout)
    except Exception as exc:
        print(json.dumps({"errcode": -1, "errmsg": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("errcode") == 0 and result.get("media_id"):
        print(f"media_id={result['media_id']}")
    return 0 if result.get("errcode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
