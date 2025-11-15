# export_to_csv.py
import os
import json
from datetime import datetime, timedelta

import pandas as pd
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://korlofxtailwltuhydya.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0OTE4NTEsImV4cCI6MjA3ODA2Nzg1MX0.Z0obqdlv31ce66ks6dCpZzEDLGLQ1D0A3QcltowP9xc",
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_laptops():
    res = supabase.table("laptops").select("*").execute()
    return res.data or []

def fetch_search_logs(limit=1000):
    # lấy nhiều log nhất có thể
    res = (
        supabase.table("search_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

def fetch_click_logs(limit=2000):
    res = (
        supabase.table("laptop_click_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def main():
    laptops = fetch_laptops()
    search_logs = fetch_search_logs()
    click_logs = fetch_click_logs()

    # map laptop_id -> row để lấy feature
    laptop_map = {str(l["id"]): l for l in laptops}

    rows = []

    for log in search_logs:
        raw_query = log.get("raw_query") or ""
        parsed_struct = log.get("parsed_struct") or {}
        result_ids = log.get("result_ids") or []
        user_id = log.get("user_id")
        created_at = parse_iso(log.get("created_at"))

        # lấy budget / usage / brand từ parsed_struct (do bạn tự parse bên Flask)
        budget = parsed_struct.get("budget")
        # bạn có thể dùng "usages" kiểu list => an toàn hơn
        usages = parsed_struct.get("usages") or parsed_struct.get("usage")
        brand_pref = parsed_struct.get("brand")

        # tìm click tương ứng (lọc theo user + thời gian gần query)
        # tạm: click trong 10 phút sau query sẽ coi là liên quan
        related_clicks = []
        if user_id:
            for c in click_logs:
                if c.get("user_id") == user_id:
                    c_time = parse_iso(c.get("created_at"))
                    if created_at and c_time:
                        if timedelta(minutes=0) <= (c_time - created_at) <= timedelta(minutes=10):
                            related_clicks.append(c)

        clicked_ids = {c.get("laptop_id") for c in related_clicks if c.get("laptop_id")}

        # với mỗi laptop mà bạn đã trả về -> tạo 1 dòng
        for lap_id in result_ids:
            lap = laptop_map.get(str(lap_id))
            if not lap:
                continue

            price = lap.get("price")
            ram_gb = lap.get("ram_gb")
            storage_gb = lap.get("storage_gb")
            lap_brand = (lap.get("brand") or "").lower()
            cpu = lap.get("cpu") or ""
            gpu = lap.get("gpu") or ""

            clicked = 1 if str(lap_id) in clicked_ids else 0

            rows.append(
                {
                    "user_id": user_id,
                    "query": raw_query,
                    "budget": budget,
                    "pref_brand": brand_pref,
                    "usage": usages,  # có thể là list hoặc str
                    "laptop_id": lap_id,
                    "laptop_brand": lap_brand,
                    "price": price,
                    "ram_gb": ram_gb,
                    "storage_gb": storage_gb,
                    "cpu": cpu,
                    "gpu": gpu,
                    "clicked": clicked,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("train_laptops.csv", index=False)
    print("Đã xuất train_laptops.csv với", len(df), "dòng")

if __name__ == "__main__":
    main()
