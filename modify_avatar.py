#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.parse
import urllib.request


MODIFY_AVATAR_URL = "https://alarm.im.qihoo.net/robot/avatar/modify"


def modify_avatar(appid, secret, avatar, timeout=30):
    appid = appid.strip()
    secret = secret.strip()
    avatar = avatar.strip()

    query = urllib.parse.urlencode({
        "appid": appid,
        "secret": secret,
    })
    url = f"{MODIFY_AVATAR_URL}?{query}"
    body = json.dumps({"avatar": avatar}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def main():
    parser = argparse.ArgumentParser(description="修改机器人头像，avatar 为上传图片后得到的 media_id")
    parser.add_argument("--appid", required=True, help="应用 appid")
    parser.add_argument("--secret", required=True, help="应用 secret")
    parser.add_argument("--avatar", required=True, help="头像图片 media_id")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间，默认 30 秒")
    args = parser.parse_args()

    try:
        result = modify_avatar(args.appid, args.secret, args.avatar, args.timeout)
    except Exception as exc:
        print(json.dumps({"errcode": -1, "errmsg": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("errcode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
