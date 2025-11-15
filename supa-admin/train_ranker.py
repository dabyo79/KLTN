# train_ranker.py
import os
import json
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
import pandas as pd
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# === cấu hình Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://korlofxtailwltuhydya.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0OTE4NTEsImV4cCI6MjA3ODA2Nzg1MX0.Z0obqdlv31ce66ks6dCpZzEDLGLQ1D0A3QcltowP9xc",
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === hàm detect brand và parse query giống app.py ===
BRAND_ALIASES = {
    "apple": ["apple", "mac", "macbook", "mac book"],
    "dell": ["dell", "del"],
    "hp": ["hp", "h p", "hpp"],
    "lenovo": ["lenovo", "leno", "leno vo", "lenoovo"],
    "asus": ["asus", "asuss", "aus", "vivobook", "assus"],
    "acer": ["acer", "a cer"],
    "msi": ["msi", "m s i"],
}

def detect_brand(t: str):
    t = t.lower()
    for canonical, variants in BRAND_ALIASES.items():
        for v in variants:
            if v in t:
                return canonical
    return None

def parse_user_query_to_struct(text: str):
    import re
    if not text:
        return {"budget": None, "usage": [], "brand": None, "raw": text}

    t = text.lower().strip()
    brand = detect_brand(t)
    budget = None

    m = re.search(r"dưới\s+(\d+[.,]?\d*)\s*(triệu|tr|m)?", t)
    if m:
        num = m.group(1).replace(",", ".")
        unit = m.group(2)
        val = float(num)
        budget = int(val * 1_000_000) if unit else int(val)

    if budget is None:
        m = re.search(r"(khoảng|tầm)\s+(\d+[.,]?\d*)\s*(triệu|tr|m)?", t)
        if m:
            val = float(m.group(2).replace(",", "."))
            unit = m.group(3)
            budget = int(val * 1_000_000) if unit else int(val)

    if budget is None:
        m = re.search(r"(\d+[.,]?\d*)\s*(triệu|tr|m)\b", t)
        if m:
            val = float(m.group(1).replace(",", "."))
            budget = int(val * 1_000_000)

    if budget is None:
        m = re.search(r"\b(\d{6,9})\b", t)
        if m:
            budget = int(m.group(1))

    usages = []
    if "học" in t or "sinh viên" in t:
        usages.append("study")
    if "thiết kế" in t or "đồ họa" in t or "design" in t:
        usages.append("design")
    if "game" in t or "chơi game" in t:
        usages.append("gaming")
    if "văn phòng" in t or "office" in t:
        usages.append("office")
    if "lập trình" in t or "làm việc" in t:
        usages.append("work")

    return {
        "budget": budget,
        "usage": usages,
        "brand": brand,
        "raw": text,
    }

def build_features_from_struct_and_laptop(struct: dict, lap: dict):
    # giống build_features_for_items trong app.py
    budget = struct.get("budget") or 0
    usages = struct.get("usage") or []
    if isinstance(usages, str):
        usages = [usages]
    pref_brand = (struct.get("brand") or "").lower()

    price = float(lap.get("price") or 0)
    lap_brand = (lap.get("brand") or "").lower()
    ram_gb = lap.get("ram_gb") or 0
    storage_gb = lap.get("storage_gb") or 0

    row = {
        "abs_price_diff": abs(price - budget),
        "brand_match": 1 if pref_brand and pref_brand == lap_brand else 0,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "price": price,
        "usage_study": 1 if "study" in usages else 0,
        "usage_design": 1 if "design" in usages else 0,
        "usage_gaming": 1 if "gaming" in usages else 0,
        "usage_office": 1 if "office" in usages else 0,
        "usage_work": 1 if "work" in usages else 0,
    }
    return row

def main():
    # 1) lấy laptop để join
    laps_res = supabase.table("laptops").select("*").execute()
    laptops = {str(x["id"]): x for x in (laps_res.data or [])}

    # 2) lấy search_logs
    logs_res = (
        supabase.table("search_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(10000)
        .execute()
    )
    search_logs = logs_res.data or []

    # 3) lấy click để đánh nhãn
    click_res = (
        supabase.table("laptop_click_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(20000)
        .execute()
    )
    click_logs = click_res.data or []

    # chuyển click thành set cho dễ tra
    clicked_pairs = set()
    for c in click_logs:
        u = c.get("user_id")
        lid = str(c.get("laptop_id"))
        if u and lid:
            clicked_pairs.add((u, lid))

    rows = []
    for log in search_logs:
        user_id = log.get("user_id")
        raw_query = log.get("raw_query") or ""
        parsed = log.get("parsed_struct")
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = parse_user_query_to_struct(raw_query)
        if not parsed:
            parsed = parse_user_query_to_struct(raw_query)

        result_ids = log.get("result_ids") or []
        # result_ids trong supabase có thể là json string
        if isinstance(result_ids, str):
            try:
                result_ids = json.loads(result_ids)
            except Exception:
                result_ids = []

        for lap_id in result_ids:
            lap = laptops.get(str(lap_id))
            if not lap:
                continue

            feats = build_features_from_struct_and_laptop(parsed, lap)

            # nhãn: nếu user có click đúng cặp (user, lap) thì =1
            label = 1 if user_id and (user_id, str(lap_id)) in clicked_pairs else 0

            feats["label"] = label
            rows.append(feats)

    if not rows:
      print("Không tạo được dữ liệu huấn luyện.")
      return

    df = pd.DataFrame(rows)
    print("Dataset shape:", df.shape)
    print(df["label"].value_counts())

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    out = {
        "model": model,
        "feature_cols": list(X.columns),
    }
    joblib.dump(out, "laptop_ranker.pkl")
    print("✅ Đã lưu model vào laptop_ranker.pkl")

if __name__ == "__main__":
    main()
