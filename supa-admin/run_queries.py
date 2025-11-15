import json
import requests

# URL Flask của bạn
BASE_URL = "http://192.168.100.237:5000/api/recommend"

# 10 câu truy vấn mẫu (bạn đổi theo bài của bạn)
QUERIES = [
    "laptop dưới 15tr cho sinh viên",
    "máy chơi game có card rời",
    "asus văn phòng",
    "macbook cho design",
    "laptop mỏng nhẹ đi học",
    "hp dưới 20tr",
    "laptop code",
    "laptop học online",
    "gaming 16gb ram",
    "dell văn phòng",
]

USER_ID = "8378725d-82e9-4a3a-b3c4-4d0f9e2a1a99"  # thay bằng user của bạn

results = []

for q in QUERIES:
    payload = {
        "user_id": USER_ID,
        "query": q,
        "device": "python-test",
    }

    try:
        resp = requests.post(BASE_URL, json=payload, timeout=5)

        # thử parse JSON
        try:
            data = resp.json()
        except Exception as e:
            # server trả không phải json (html, rỗng, 500...)
            data = {
                "error": str(e),
                "status": resp.status_code,
                "raw": resp.text,
            }

    except Exception as e:
        # lỗi kết nối, timeout, sai IP...
        data = {"error": str(e)}

    # lưu lại để sau đối chiếu
    results.append(
        {
            "query": q,
            "response": data,
        }
    )

# ghi ra file
with open("results_ml.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Đã gọi xong 10 query và lưu vào results_ml.json")
