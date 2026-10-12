"""
一次性工具：通过 Wikimedia Commons API 搜索并下载猫狗测试样例图
"""
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")
UA = {"User-Agent": "Mozilla/5.0 dog-cat-demo/1.0"}
# 每类搜索关键词，各取前若干张
SEARCHES = [
    ("cat", "domestic cat sitting"),
    ("cat", "tabby cat portrait"),
    ("dog", "golden retriever dog"),
    ("dog", "labrador retriever dog"),
]
PER_SEARCH = 3


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def search_files(query: str, limit: int) -> list:
    """搜索图片文件，返回 File: 标题列表"""
    data = api_get({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6", "gsrlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url|size",
    })
    pages = data.get("query", {}).get("pages", {})
    results = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        url, w, h = info.get("url"), info.get("width", 0), info.get("height", 0)
        # URL 可能带 ?utm_* 查询串，先去掉再判断扩展名
        path = url.split("?", 1)[0].lower() if url else ""
        # 过滤小图和非 jpg/png
        if url and w >= 600 and h >= 600 and path.endswith((".jpg", ".jpeg", ".png")):
            results.append((p["title"], url))
    return results[:limit]


def download(url: str, dest: str) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        is_image = data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG"
        if len(data) > 20000 and is_image:
            with open(dest, "wb") as f:
                f.write(data)
            return True
        print(f"  BAD 内容校验失败: {url}")
        return False
    except Exception as e:
        print(f"  FAIL {e}")
        return False


def main():
    os.makedirs(BASE, exist_ok=True)
    counter = {"cat": 0, "dog": 0}
    for label, query in SEARCHES:
        print(f"搜索 [{label}] {query} ...")
        try:
            files = search_files(query, PER_SEARCH)
        except Exception as e:
            print(f"  搜索失败: {e}")
            continue
        for title, url in files:
            if counter[label] >= 6:
                break
            counter[label] += 1
            ext = ".png" if url.split("?", 1)[0].lower().endswith(".png") else ".jpg"
            dest = os.path.join(BASE, label, f"{label}{counter[label]}{ext}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            ok = download(url, dest)
            print(f"  {'OK  ' if ok else 'FAIL'} {label}{counter[label]} <- {title}")
            time.sleep(1.5)  # 避免触发 429 限流
    for label in ("cat", "dog"):
        n = len(os.listdir(os.path.join(BASE, label))) if os.path.isdir(os.path.join(BASE, label)) else 0
        print(f"{label}: {n} 张")


if __name__ == "__main__":
    main()
