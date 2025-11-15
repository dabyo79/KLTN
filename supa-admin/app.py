import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from supabase import create_client, Client
from datetime import datetime
from flask_cors import CORS
from pathlib import Path
import json
import re
import joblib
import pandas as pd
from uuid import UUID
from collections import Counter

# ========= CẤU HÌNH SUPABASE =========
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://korlofxtailwltuhydya.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0OTE4NTEsImV4cCI6MjA3ODA2Nzg1MX0.Z0obqdlv31ce66ks6dCpZzEDLGLQ1D0A3QcltowP9xc",
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

# ========== LOAD MODEL ==========

try:
    RANKER = joblib.load("laptop_ranker.pkl")
    ML_MODEL = RANKER["model"]
    FEATURE_COLS = RANKER["feature_cols"]
    print("Loaded ML model for ranking")
except Exception as e:
    ML_MODEL = None
    FEATURE_COLS = []
    print("Không load được model, sẽ dùng rule thuần:", e)


def build_features_for_items(struct, items):
    budget = struct.get("budget") or 0
    usages = struct.get("usage") or []

    if isinstance(usages, str):
        usages = [usages]
    pref_brand = (struct.get("brand") or "").lower()

    rows = []
    for it in items:
        price = float(it.get("price") or 0)
        lap_brand = (it.get("brand") or "").lower()

        # ép về int vì supabase có thể trả "512"
        ram_raw = it.get("ram_gb") or 0
        try:
            ram_gb = int(ram_raw)
        except Exception:
            ram_gb = 0

        storage_raw = it.get("storage_gb") or 0
        try:
            storage_gb = int(storage_raw)
        except Exception:
            storage_gb = 0

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
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS]



# ====== HÀM PHÂN TÍCH QUERY (tạm) ======
BRAND_ALIASES = {
    "apple": ["apple", "mac", "macbook", "mac book"],
    "dell": ["dell", "del"],
    "hp": ["hp", "h p", "hpp"],
    "lenovo": ["lenovo", "leno", "leno vo", "lenoovo"],
    "asus": ["asus", "asuss", "aus", "vivobook", "assus"],
    "acer": ["acer", "a cer"],
    "msi": ["msi", "m s i"],
}
DISCRETE_GPU_KEYWORDS = [
    "rtx", "gtx",     # nvidia gaming
    "quadro",         # nvidia workstation
    "radeon rx",      # amd gaming
    "rx ",            # ví dụ "RX 6600"
    "t500", "t1000",  # mấy dòng nvidia mobile
]

def has_discrete_gpu(gpu_text: str) -> bool:
    if not gpu_text:
        return False
    g = gpu_text.lower()

    # iGPU phổ biến
    if "iris xe" in g or "intel uhd" in g or g.strip() == "radeon graphics":
        return False

    return any(k in g for k in DISCRETE_GPU_KEYWORDS)



def detect_brand(t: str):
    t = t.lower()
    for canonical, variants in BRAND_ALIASES.items():
        for v in variants:
            if v in t:
                return canonical
    return None


def parse_user_query_to_struct(text: str):
    if not text:
        return {"budget": None, "usage": None, "brand": None, "raw": text}

    t = text.lower().strip()
    brand = detect_brand(t)
    budget = None

    m = re.search(r"dưới\s+(\d+[.,]?\d*)\s*(triệu|tr|m)?", t)
    if m:
        num = m.group(1).replace(",", ".")
        unit = m.group(2)
        val = float(num)
        if unit in ("triệu", "tr", "m"):
            budget = int(val * 1_000_000)
        else:
            budget = int(val)

    if budget is None:
        m = re.search(r"(khoảng|tầm|tầm khoảng)\s+(\d+[.,]?\d*)\s*(triệu|tr|m)?", t)
        if m:
            val = float(m.group(2).replace(",", "."))
            unit = m.group(3)
            if unit in ("triệu", "tr", "m"):
                budget = int(val * 1_000_000)
            else:
                budget = int(val)

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
    avoid_dgpu = False   # 👈 thêm cờ này
    needs_dgpu = False
    # học
    if "học" in t or "hoc" in t or "sinh viên" in t or "sinh vien" in t or "sv" in t:
        usages.append("study")
        # máy học thường không cần card rời
        if not needs_dgpu:
            avoid_dgpu = True

    # văn phòng / office
    if ("văn phòng" in t or "van phong" in t or "office" in t or
        "word" in t or "excel" in t or "powerpoint" in t):
        usages.append("office")
        # nếu chỉ nói văn phòng thì ưu tiên máy iGPU
        if not needs_dgpu:
            avoid_dgpu = True

    # làm việc / dev
    if ("làm việc" in t or "lam viec" in t or "công việc" in t or "cong viec" in t or
        "dev" in t or "lập trình" in t or "lap trinh" in t or "program" in t or "code" in t):
        usages.append("work")

    # họp online
    if "zoom" in t or "teams" in t or "meet" in t:
        if "office" not in usages:
            usages.append("office")
        if not needs_dgpu:
            avoid_dgpu = True

    # gaming
    if "game" in t or "chơi game" in t or "chơi" in t:
        usages.append("gaming")
        needs_dgpu = True
        avoid_dgpu = False   # nói game thì thôi đừng né card rời

    # user nói rõ là muốn card rời
    if ("card rời" in t or "card roi" in t or
        "vga rời" in t or "gpu rời" in t or "gpu roi" in t):
        needs_dgpu = True
        avoid_dgpu = False

    return {
        "budget": budget,
        "usage": usages,
        "brand": brand,
        "raw": text,
        "needs_dgpu": needs_dgpu,
        "avoid_dgpu": avoid_dgpu,
    }


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "laptop_recommender" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAFFIC_PATH = DATA_DIR / "traffic.json"


def load_traffic():
    if TRAFFIC_PATH.exists():
        return json.loads(TRAFFIC_PATH.read_text(encoding="utf-8"))
    return []


def save_traffic(data):
    TRAFFIC_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/api/mobile/traffic_ping", methods=["POST"])
def traffic_ping():
    payload = request.get_json(force=True) or {}
    payload["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    traffic = load_traffic()
    traffic.append(payload)
    save_traffic(traffic)
   
    return jsonify({"ok": True})


# ========== HELPER: CÁ NHÂN HÓA THEO USER ==========
def get_user_preference(user_id: str):
    """
    Lấy brand, khoảng giá, và RAM mà user này hay click nhất.
    Trả về (brand_scores, price_bucket_scores, ram_scores)
    """
    if not user_id:
        return {}, {}, {}

    # lấy 10 click gần nhất
    click_res = (
        supabase.table("laptop_click_logs")
        .select("laptop_id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    clicks = click_res.data or []
    if not clicks:
        return {}, {}, {}

    lap_ids = [c["laptop_id"] for c in clicks if c.get("laptop_id")]
    lap_ids = list({lid for lid in lap_ids})

    laps_res = (
        supabase.table("laptops")
        .select("id,brand,price,ram_gb")
        .in_("id", lap_ids)
        .execute()
    )
    laps = laps_res.data or []

    brand_scores = {}
    price_bucket_scores = {}
    ram_scores = {}

    def bucket(p):
        p = p or 0
        if p < 10_000_000:
            return "<10"
        elif p < 15_000_000:
            return "10-15"
        elif p < 20_000_000:
            return "15-20"
        else:
            return ">20"

    for lap in laps:
        # brand
        b = (lap.get("brand") or "").lower()
        if b:
            brand_scores[b] = brand_scores.get(b, 0) + 1

        # price bucket
        pb = bucket(lap.get("price") or 0)
        price_bucket_scores[pb] = price_bucket_scores.get(pb, 0) + 1

        # ram
        rg = lap.get("ram_gb")
        if rg is not None:
            ram_scores[rg] = ram_scores.get(rg, 0) + 1

    return brand_scores, price_bucket_scores, ram_scores



def price_bucket_of_item(item):
    p = item.get("price") or 0
    if p < 10_000_000:
        return "<10"
    elif p < 15_000_000:
        return "10-15"
    elif p < 20_000_000:
        return "15-20"
    else:
        return ">20"


# ========== SERVE REACT ==========
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(app.static_folder, "static"), path)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")


# ========== API GỢI Ý LAPTOP ==========
def to_uuid_or_none(val):
    try:
        return str(UUID(str(val)))
    except Exception:
        return None

def has_explicit_filters(filters: dict) -> bool:
    if not filters:
        return False
    if filters.get("price"):
        if filters["price"].get("min") is not None or filters["price"].get("max") is not None:
            return True
    if filters.get("brand"):
        if isinstance(filters["brand"], list) and filters["brand"]:
            return True
        if isinstance(filters["brand"], str) and filters["brand"].strip():
            return True
    if filters.get("ram"):
        if isinstance(filters["ram"], list) and filters["ram"]:
            return True
        if isinstance(filters["ram"], int):
            return True
    if filters.get("gpu"):
        return True
    # bạn có thể thêm các filter khác ở đây
    return False


def apply_explicit_filters(laptops: list, filters: dict) -> list:
    if not filters:
        return laptops

    price_f = filters.get("price") or {}
    brand_f = filters.get("brand")
    ram_f = filters.get("ram")
    gpu_f = filters.get("gpu")  # ví dụ: "discrete" | "integrated"

    out = []
    for lap in laptops:
        ok = True

        # giá
        if price_f:
            p = float(lap.get("price") or 0)
            minp = price_f.get("min")
            maxp = price_f.get("max")
            if minp is not None and p < float(minp):
                ok = False
            if maxp is not None and p > float(maxp):
                ok = False

        # brand
        if ok and brand_f:
            lb = (lap.get("brand") or "").lower()
            if isinstance(brand_f, list):
                brand_f_norm = [b.lower() for b in brand_f]
                if lb not in brand_f_norm:
                    ok = False
            else:  # string
                if lb != str(brand_f).lower():
                    ok = False

        # ram
        if ok and ram_f:
            lap_ram = lap.get("ram_gb")
            if isinstance(ram_f, list):
                if lap_ram not in ram_f:
                    ok = False
            else:  # int
                if lap_ram != ram_f:
                    ok = False

        # gpu
        if ok and gpu_f:
            if gpu_f == "discrete":
                if not has_discrete_gpu(lap.get("gpu", "")):
                    ok = False
            elif gpu_f == "integrated":
                if has_discrete_gpu(lap.get("gpu", "")):
                    ok = False

        if ok:
            out.append(lap)

    return out

@app.route("/api/recommend", methods=["POST"])
    
def recommend():
    data = request.get_json() or {}

    user_id = data.get("user_id")
    raw_query = data.get("query", "") or ""
    device = data.get("device", "web")
    

    # 👇 nhận filter từ app
    filter_brand = (data.get("filter_brand") or "").strip().lower() or None
    min_price = data.get("min_price")
    max_price = data.get("max_price")

    # parse từ câu gõ
    struct = parse_user_query_to_struct(raw_query)


    # 👇 THÊM 3 DÒNG NÀY ĐỂ LOG RA ADMIN THẤY FILTER
    if filter_brand:
        struct["filter_brand"] = filter_brand
    if min_price is not None:
        struct["min_price"] = min_price
    if max_price is not None:
        struct["max_price"] = max_price

    brand_from_query = struct.get("brand")
    budget = struct.get("budget")
    needs_dgpu = struct.get("needs_dgpu", False)

    # lấy laptop còn hàng
    res = supabase.table("laptops").select("*").eq("in_stock", True).execute()
    laptops = res.data or []

    # ========== 1. ÁP FILTER TỪ APP TRƯỚC ==========
    # filter brand do user chọn trên app
    if filter_brand:
        laptops = [
            lap for lap in laptops
            if (lap.get("brand") or "").lower().startswith(filter_brand)
        ]

    # filter giá do user chọn trên app
    def ok_price(lap):
        p = float(lap.get("price") or 0)
        if min_price is not None and p < float(min_price):
            return False
        if max_price is not None and p > float(max_price):
            return False
        return True

    if min_price is not None or max_price is not None:
        laptops = [lap for lap in laptops if ok_price(lap)]

    # ========== 2. NẾU KHÔNG CÓ QUERY, CHỈ LỌC THÔI ==========
    # -> trả đúng kết quả lọc, không cần ML
    if raw_query.strip() == "":
        ranked = sorted(laptops, key=lambda x: float(x.get("price") or 0))
        # log lại như cũ
        insert_data = {
            "raw_query": raw_query,
            "parsed_struct": struct,
            "result_ids": [item["id"] for item in ranked],
            "device": device,
        }
        if user_id:
            check = (
                supabase.table("profiles")
                .select("id")
                .eq("id", user_id)
                .execute()
            )
            if check.data:
                insert_data["user_id"] = user_id
        supabase.table("search_logs").insert(insert_data).execute()

        return jsonify({
            "status": "ok",
            "used_ml": False,
            "items": ranked,
        })

    # ========== 3. CÓ QUERY -> dùng logic cũ + thêm filter ==========
    # lọc brand theo query (nếu query cũng nói brand)
    if brand_from_query:
        laptops = [
            lap for lap in laptops
            if (lap.get("brand") or "").lower().startswith(brand_from_query)
        ]

    # lọc theo budget suy ra từ query (cho rộng 1.2)
    if budget:
        tmp = [lap for lap in laptops if float(lap.get("price") or 0) <= float(budget) * 1.2]
        if tmp:
            laptops = tmp

    # bắt buộc card rời nếu query nói gaming
    if needs_dgpu:
        only_dgpu = [lap for lap in laptops if has_discrete_gpu(lap.get("gpu", ""))]
        if only_dgpu:
            laptops = only_dgpu

    # rule score nhẹ
    def score(item):
        s = 0
        price = float(item.get("price") or 0)
        if budget:
            diff = abs(price - float(budget))
            if diff < 2_000_000:
                s += 20
            elif diff < 5_000_000:
                s += 10

        usages = struct.get("usage") or []
        if isinstance(usages, str):
            usages = [usages]
        if "design" in usages and (item.get("ram_gb") or 0) >= 16:
            s += 10
        return s

    ranked = sorted(laptops, key=score, reverse=True)

    # ===== ML =====
    used_ml = False
    if ML_MODEL is not None and ranked:
        feat_df = build_features_for_items(struct, ranked)
        scores = ML_MODEL.predict_proba(feat_df)[:, 1]
        for i, item in enumerate(ranked):
            item["_ml_score"] = float(scores[i])
        ranked = sorted(ranked, key=lambda x: x["_ml_score"], reverse=True)
        used_ml = True

    # ===== cá nhân hóa =====
    query_usages = struct.get("usage") or []
    if isinstance(query_usages, str):
        query_usages = [query_usages]

    brand_pref, price_pref, ram_pref = get_user_preference(user_id)

    if brand_pref or price_pref or ram_pref or query_usages:
        for item in ranked:
            bonus = 0.0

            # 1. brand từ lịch sử
            item_brand = (item.get("brand") or "").lower()
            if item_brand in brand_pref:
                bonus += 0.5 * brand_pref[item_brand]

            # 2. price từ lịch sử
            bucket = price_bucket_of_item(item)
            if bucket in price_pref:
                bonus += 0.3 * price_pref[bucket]

            # 3. RAM từ lịch sử
            item_ram = item.get("ram_gb")
            if item_ram is not None and item_ram in ram_pref:
                bonus += 0.2 * ram_pref[item_ram]

            # 4. usage từ câu hỏi hiện tại
            if "gaming" in query_usages:
                if has_discrete_gpu(item.get("gpu", "")):
                    bonus += 0.6
                elif item.get("gpu"):
                    bonus += 0.1
                if (item.get("ram_gb") or 0) >= 16:
                    bonus += 0.2

            if "design" in query_usages:
                if (item.get("ram_gb") or 0) >= 16:
                    bonus += 0.3
                if has_discrete_gpu(item.get("gpu", "")):
                    bonus += 0.2

            if "study" in query_usages or "office" in query_usages:
                if (item.get("price") or 0) < 15_000_000:
                    bonus += 0.1
                if item.get("in_stock", True):
                    bonus += 0.05

            item["_user_boost"] = bonus

        # sắp xếp lại theo ML + boost
        ranked = sorted(
            ranked,
            key=lambda x: (x.get("_ml_score", 0.0) + x.get("_user_boost", 0.0)),
            reverse=True
        )

    # log
    insert_data = {
        "raw_query": raw_query,
        "parsed_struct": struct,
        "result_ids": [item["id"] for item in ranked],
        "device": device,
    }
    if user_id:
        check = (
            supabase.table("profiles")
            .select("id")
            .eq("id", user_id)
            .execute()
        )
        if check.data:
            insert_data["user_id"] = user_id
    supabase.table("search_logs").insert(insert_data).execute()

    return jsonify({
        "status": "ok",
        "used_ml": used_ml,
        "items": ranked,
    })





# ========== WEB ADMIN ==========
@app.route("/admin")
def admin_home():
    return render_template("admin.html")


@app.route("/admin/stats")
def admin_stats():
    res = (
        supabase.table("search_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    logs = res.data or []
    return render_template("admin_stats.html", logs=logs)


@app.route("/admin/api/stats_json")
def stats_json():
    logs_res = (
        supabase.table("search_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(500)
        .execute()
    )
    logs = logs_res.data or []

    counts_by_user = {}
    for row in logs:
        uid = row.get("user_id")
        if not uid:
            continue
        counts_by_user[uid] = counts_by_user.get(uid, 0) + 1

    top_user_ids = sorted(
        counts_by_user.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    profiles_map = {}
    if top_user_ids:
        ids_only = [u[0] for u in top_user_ids]
        prof_res = (
            supabase.table("profiles")
            .select("id,full_name,avatar_url")
            .in_("id", ids_only)
            .execute()
        )
        for p in prof_res.data or []:
            profiles_map[p["id"]] = {
                "full_name": p.get("full_name"),
                "avatar_url": p.get("avatar_url"),
            }

    top_users_from_logs = []
    for uid, total in top_user_ids:
        prof = profiles_map.get(uid, {})
        top_users_from_logs.append({
            "user_id": uid,
            "total_search": total,
            "full_name": prof.get("full_name") or uid,
            "avatar_url": prof.get("avatar_url"),
        })

    KNOWN_BRANDS = ["apple", "dell", "hp", "lenovo", "asus", "acer", "msi"]
    brand_counts = {b: 0 for b in KNOWN_BRANDS}

    for row in logs:
        parsed = row.get("parsed_struct") or {}
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = {}
        b = (parsed.get("brand") or "").lower()
        if b in brand_counts:
            brand_counts[b] += 1

    brand_from_logs = [
        {"brand": b, "total": brand_counts[b]} for b in KNOWN_BRANDS
    ]

    click_res = (
        supabase.table("laptop_click_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )
    click_rows = click_res.data or []

    click_count = {}
    for row in click_rows:
        lap_id = row.get("laptop_id")
        if not lap_id:
            continue
        click_count[lap_id] = click_count.get(lap_id, 0) + 1

    top_click_ids = sorted(
        click_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    top_clicked = []
    for lap_id, total in top_click_ids:
        lap_res = (
            supabase.table("laptops")
            .select("name,image_url")
            .eq("id", lap_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if lap_res:
            top_clicked.append({
                "laptop_id": lap_id,
                "name": lap_res[0].get("name"),
                "image_url": lap_res[0].get("image_url"),
                "total_click": total,
            })
        else:
            top_clicked.append({
                "laptop_id": lap_id,
                "name": f"#{lap_id[:6]}",
                "image_url": None,
                "total_click": total,
            })

    traffic_logs = load_traffic()
    return jsonify({
        "logs": logs,
        "top_search_users": top_users_from_logs,
        "brand_from_logs": brand_from_logs,
        "top_clicked_laptops": top_clicked,
        "traffic_logs": traffic_logs,
    })


# ========== API LOG CLICK ==========
@app.route("/api/log_click", methods=["POST"])
def log_click():
    data = request.get_json() or {}

    user_id = data.get("user_id", "guest_user")
    laptop_id = data.get("laptop_id")
    duration_ms = data.get("duration_ms")

    if not laptop_id:
        return jsonify({"error": "missing laptop_id"}), 400

    row = {
        "user_id": user_id,
        "laptop_id": laptop_id,
    }

    if duration_ms is not None:
        row["duration_ms"] = int(duration_ms)

    supabase.table("laptop_click_logs").insert(row).execute()

    return jsonify({"status": "ok"})


@app.route("/admin/api/user_stats")
def user_stats():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "missing user_id"}), 400

    rec_res = (
        supabase.table("search_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    recent_recommends = rec_res.data or []

    click_res = (
        supabase.table("laptop_click_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    click_rows = click_res.data or []

    counter = Counter()
    for r in click_rows:
        lid = r.get("laptop_id")
        if lid:
            counter[lid] += 1

    top_clicked = []
    for lid, total in counter.most_common(10):
        lap = (
            supabase.table("laptops")
            .select("name,price,image_url")
            .eq("id", lid)
            .limit(1)
            .execute()
            .data
        )
        if lap:
            lap = lap[0]
            top_clicked.append({
                "laptop_id": lid,
                "name": lap.get("name"),
                "price": lap.get("price"),
                "image_url": lap.get("image_url"),
                "total": total,
            })
        else:
            top_clicked.append({
                "laptop_id": lid,
                "name": lid,
                "price": None,
                "image_url": None,
                "total": total,
            })

    view_rows = (
        supabase.table("laptop_click_logs")
        .select("laptop_id,duration_ms,created_at")
        .eq("user_id", user_id)
        .not_.is_("duration_ms", "null")
        .order("duration_ms", desc=True)
        .limit(10)
        .execute()
        .data or []
    )

    longest_stay = []
    for row in view_rows:
        lid = row["laptop_id"]
        lap = (
            supabase.table("laptops")
            .select("name,image_url")
            .eq("id", lid)
            .limit(1)
            .execute()
            .data
        )
        if lap:
            lap = lap[0]
            longest_stay.append({
                "laptop_id": lid,
                "name": lap.get("name"),
                "image_url": lap.get("image_url"),
                "duration_ms": row.get("duration_ms") or 0,
                "created_at": row.get("created_at"),
            })
        else:
            longest_stay.append({
                "laptop_id": lid,
                "name": lid,
                "image_url": None,
                "duration_ms": row.get("duration_ms") or 0,
                "created_at": row.get("created_at"),
            })

    carts = []

    price_buckets = {
        "< 10tr": 0,
        "10-15tr": 0,
        "15-20tr": 0,
        "> 20tr": 0,
    }
    for r in click_rows:
        lid = r.get("laptop_id")
        if not lid:
            continue
        lap = (
            supabase.table("laptops")
            .select("price")
            .eq("id", lid)
            .limit(1)
            .execute()
            .data
        )
        if not lap:
            continue
        price = lap[0].get("price") or 0
        if price < 10_000_000:
            price_buckets["< 10tr"] += 1
        elif price < 15_000_000:
            price_buckets["10-15tr"] += 1
        elif price < 20_000_000:
            price_buckets["15-20tr"] += 1
        else:
            price_buckets["> 20tr"] += 1

    return jsonify({
        "recent_recommends": recent_recommends,
        "top_clicked": top_clicked,
        "longest_stay": longest_stay,
        "carts": carts,
        "price_buckets": price_buckets,
    })
from flask import request, jsonify

@app.route("/admin/api/flag_chat", methods=["POST"])
def flag_chat():
    data = request.get_json(force=True)
    # TODO: ghi vào Supabase hoặc file/log
    # Ví dụ Supabase:
    # supabase.table("support_flags").insert({
    #   "sender_id": data.get("sender_id"),
    #   "last_user_message": data.get("last_user_message"),
    #   "reason": data.get("reason"),
    #   "ts": data.get("ts"),
    #   "handled": False
    # }).execute()
    print("FLAG_CHAT:", data)  # debug
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
