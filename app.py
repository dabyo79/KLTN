import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone, date
import random
import smtplib
from email.mime.text import MIMEText
from flask_cors import CORS
from pathlib import Path
import json
import re
import joblib
import pandas as pd
from uuid import UUID
from collections import Counter, defaultdict
import requests   
import time
import sys
import hashlib
import hmac
import json
import time
import requests
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

# ========= CẤU HÌNH SUPABASE =========
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://korlofxtailwltuhydya.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0OTE4NTEsImV4cCI6MjA3ODA2Nzg1MX0.Z0obqdlv31ce66ks6dCpZzEDLGLQ1D0A3QcltowP9xc",
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjQ5MTg1MSwiZXhwIjoyMDc4MDY3ODUxfQ.khEKIZN8a7QVlKcjdtB_KPo6T_QP-T3wkhRcIP0wYKM"
)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

ORDER_STATUSES = [
    "Chờ xác nhận",
    "Chờ lấy hàng",
    "Chờ giao hàng",
    "Hoàn thành",
    "Đã hủy",
    "Trả hàng",
]
ZP_APP_ID = 554
ZP_KEY1 = "8NdU5pG5R2spGHGhyO99HN1OhD8IQJBn"
ZP_ENDPOINT = "https://sandbox.zalopay.com.vn/v001/tpe/createorder"
# ========== CẤU HÌNH SMTP ĐỂ GỬI OTP ==========
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "huydao2k3@gmail.com"          # gmail gửi OTP
SMTP_PASS = "jaxpecuhaqxsjcav"  # app password / SMTP password
AUTH_ADMIN_URL = f"{SUPABASE_URL}/auth/v1/admin"
app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

def log_search(
    *,
    raw_query=None,
    parsed_struct=None,
    result_ids=None,
    device=None,
    user_id=None,
    brand=None,
    usage=None,
    budget=None,
    min_price=None,
    max_price=None,
    query_type=None,
    topk=None,
    latency_ms=None,
    source_model=None,
):
    """Ghi log vào bảng search_logs."""
    payload = {
        "raw_query": raw_query,
        "parsed_struct": parsed_struct,
        "result_ids": result_ids,
        "device": device,
        "brand": brand,
        "usage": usage,
        "budget": budget,
        "user_id": user_id,
        "min_price": min_price,
        "max_price": max_price,
        "query_type": query_type,
        "topk": topk,
        "latency_ms": latency_ms,
        "source_model": source_model,
    }

    # bỏ key None để log sạch
    clean = {k: v for k, v in payload.items() if v is not None}

    try:
        supabase.table("search_logs").insert(clean).execute()
    except Exception as e:
        # không để lỗi log làm chết API recommend
        print("log_search error:", e)

# ========== API GỬI OTP RESET MẬT KHẨU ==========
def send_email(to_email: str, subject: str, body: str):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

@app.route("/api/request_reset_otp", methods=["POST"])
def request_reset_otp():
    data = request.get_json(force=True) or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "missing email"}), 400

    # random OTP 6 số
    otp = "".join(random.choices("0123456789", k=6))

    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()


    # lưu OTP vào Supabase (bảng password_reset_otps)
    supabase.table("password_reset_otps") \
        .delete().eq("email", email).execute()
    supabase.table("password_reset_otps").insert({
        "email": email,
        "otp": otp,
        "expires_at": expires_at
    }).execute()

    body = f"Mã OTP đặt lại mật khẩu của bạn là: {otp}\nCó hiệu lực trong 10 phút."
    send_email(email, "Mã OTP đặt lại mật khẩu", body)

    return jsonify({"ok": True})

@app.route("/api/reset_password_with_otp", methods=["POST"])
def reset_password_with_otp():
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    data = request.get_json(force=True) or {}
    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("new_password")

    if not email or not otp or not new_password:
        return jsonify({"ok": False, "reason": "missing_fields"}), 400

    # 1. Đọc OTP trong bảng password_reset_otps
    res = (
        supabase.table("password_reset_otps")
        .select("otp,expires_at")
        .eq("email", email)
        .eq("otp", otp)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = res.data or []
    if not rows:
        return jsonify({"ok": False, "reason": "otp_not_found"}), 400

    row = rows[0]
    expires_at_str = row.get("expires_at")

    # 2. Check hết hạn OTP
    if expires_at_str:
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at.tzinfo is None:
                # ép về UTC nếu datetime lưu là "naive"
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except Exception:
            return jsonify({"ok": False, "reason": "invalid_expires"}), 500

        now_utc = datetime.now(timezone.utc)
        if now_utc > expires_at:
            return jsonify({"ok": False, "reason": "expired"}), 400

    # 3. Gọi Supabase Auth Admin để lấy user theo email
    admin_headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }

    r1 = requests.get(
    f"{AUTH_ADMIN_URL}/users",
    headers=admin_headers,
    params={"email": f"eq.{email}"},   # 👈 quan trọng
    timeout=10,
    )
    body = r1.json()
    users = body.get("users", []) or []

# lọc lại một lần nữa theo email (phòng trường hợp api hơi quái)
    users = [u for u in users if (u.get("email") or "").lower() == email.lower()]

    if not users:
        return jsonify({"ok": False, "reason": "user_not_found"}), 400

# nếu vì lý do nào đó có >1 user cùng email → nên báo lỗi
    if len(users) > 1:
        return jsonify({"ok": False, "reason": "multiple_users_same_email"}), 500

    user_id = users[0]["id"]


    # 4. Update password qua Supabase Admin
    r2 = requests.put(
        f"{AUTH_ADMIN_URL}/users/{user_id}",
        headers={**admin_headers, "Content-Type": "application/json"},
        json={"password": new_password},
        timeout=10,
    )
    print("ADMIN_PATCH", r2.status_code, r2.text)  # 💬 LOG 2
    print("CANDIDATE_USERS", [u.get("email") for u in users])
    print("TARGET_USER", users[0].get("id"), users[0].get("email"))

    if not r2.ok:
        return jsonify(
            {"ok": False, "reason": "update_failed", "detail": r2.text}
        ), 500

    # 5. Xoá OTP sau khi dùng xong (optional)
    try:
        (
            supabase.table("password_reset_otps")
            .delete()
            .eq("email", email)
            .eq("otp", otp)
            .execute()
        )
    except Exception as e:
        print("delete otp error:", e)

    return jsonify({"ok": True})




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
# ========== LOAD K-MEANS CLUSTERS ==========
try:
    with open("laptop_kmeans_clusters.json", "r", encoding="utf-8") as f:
        LAPTOP_CLUSTER_MAP = json.load(f)
    with open("kmeans_cluster_config.json", "r", encoding="utf-8") as f:
        KMEANS_CLUSTER_CONFIG = json.load(f)
    print("Loaded KMeans clusters & config")
except Exception as e:
    LAPTOP_CLUSTER_MAP = {}
    KMEANS_CLUSTER_CONFIG = {"clusters": {}}
    print("Không load được KMeans cluster files, bỏ qua phân cụm:", e)
try:
    KMEANS_N_CLUSTERS = int(KMEANS_CLUSTER_CONFIG.get("n_clusters", 4))
except Exception:
    KMEANS_N_CLUSTERS = 4
# ========== LOAD CF ALS MODEL (COLLABORATIVE FILTERING) ==========
try:
    CF_PACKAGE = joblib.load("cf_als_model.pkl")
    CF_MODEL = CF_PACKAGE["model"]
    CF_USER2IDX = CF_PACKAGE["user2idx"]
    CF_ITEM2IDX = CF_PACKAGE["item2idx"]
    CF_IDX2ITEM = CF_PACKAGE["idx2item"]
    CF_USER_ITEM_MATRIX = CF_PACKAGE["user_item_matrix"]
    print("Loaded CF ALS model")
except Exception as e:
    CF_MODEL = None
    CF_USER2IDX = {}
    CF_ITEM2IDX = {}
    CF_IDX2ITEM = []
    CF_USER_ITEM_MATRIX = None
    print("Không load được CF ALS model:", e)

# ========== HYBRID: CONTENT + CF (ALS) ==========

def get_cf_scores_for_user_items(user_id: str, laptops: list, min_interactions: int = 3):
    """
    Trả về dict: laptop_id -> cf_score_raw (chưa scale)
    Nếu user chưa đủ tương tác / không có trong CF thì trả dict rỗng.
    """
    global CF_USER_ITEM_MATRIX  # <<< THÊM DÒNG NÀY

    if not user_id or CF_MODEL is None or CF_USER_ITEM_MATRIX is None:
        return {}

    u_idx = CF_USER2IDX.get(str(user_id))
    if u_idx is None:
        return {}

    # Đảm bảo CF_USER_ITEM_MATRIX là CSR để truy cập theo dòng (row) nhanh hơn
    if hasattr(CF_USER_ITEM_MATRIX, "tocsr"):
        try:
            # tránh convert nhiều lần nếu đã là CSR rồi
            if hasattr(CF_USER_ITEM_MATRIX, "getformat"):
                if CF_USER_ITEM_MATRIX.getformat() != "csr":
                    CF_USER_ITEM_MATRIX = CF_USER_ITEM_MATRIX.tocsr()
            else:
                CF_USER_ITEM_MATRIX = CF_USER_ITEM_MATRIX.tocsr()
        except Exception as e:
            print("Không convert được CF_USER_ITEM_MATRIX sang CSR:", e)

    # số lượng items user này đã tương tác trong ma trận
    user_row = CF_USER_ITEM_MATRIX[u_idx]
    if getattr(user_row, "nnz", 0) < min_interactions:
        # user quá “mới” -> chưa dùng CF
        return {}

    cf_scores = {}
    u_vec = CF_MODEL.user_factors[u_idx]

    for lap in laptops:
        lid = lap.get("id")
        if not lid:
            continue

        i_idx = CF_ITEM2IDX.get(str(lid))
        if i_idx is None:
            continue

        i_vec = CF_MODEL.item_factors[i_idx]
        score = float(np.dot(u_vec, i_vec))   # raw CF score
        cf_scores[str(lid)] = score

    return cf_scores



def apply_hybrid_scores(laptops: list, user_id: str, alpha: float = 0.7):
    """
    Kết hợp:
        FinalScore = alpha * ContentScore + (1 - alpha) * CFScore_norm

    - ContentScore lấy từ lap["_score"] (do ML ranker tính)
    - CFScore được min-max scale trên tập laptop đang xét
    - Chỉ áp dụng nếu user có đủ tương tác cho CF
    """
    if not user_id or CF_MODEL is None:
        return laptops

    # Lấy raw CF score cho tất cả laptop hiện có
    cf_raw = get_cf_scores_for_user_items(user_id, laptops, min_interactions=3)
    if not cf_raw:
        # user mới / không có trong CF -> giữ nguyên content-based
        return laptops

    # Gom tất cả score để scale
    vals = list(cf_raw.values())
    min_s = min(vals)
    max_s = max(vals)

    for lap in laptops:
        lid = str(lap.get("id"))
        content_score = float(lap.get("_score", 0.0))

        raw = cf_raw.get(lid)
        if raw is None or max_s == min_s:
            cf_norm = 0.0   # nếu không có cf hoặc tất cả giống nhau thì bỏ CF
        else:
            cf_norm = (raw - min_s) / (max_s - min_s)

        lap["_cf_score"] = cf_norm
        lap["_final_score"] = alpha * content_score + (1.0 - alpha) * cf_norm
        lap["_score"] = lap["_final_score"]  # để các đoạn sau dùng chung '_score'

    # sắp xếp lại theo final score
    laptops.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return laptops

def build_features_for_items(struct, items):
    budget = struct.get("budget") or 0
    usages = struct.get("usage") or []
    if isinstance(usages, str):
        usages = [usages]
    usages = [str(u).lower() for u in usages]
    pref_brand = (struct.get("brand") or "").lower()

    rows = []
    for it in items:
        price = float(it.get("price") or 0)
        lap_brand = (it.get("brand") or "").lower()

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

        purpose_slug = (it.get("purpose") or "").lower()
        purpose_match = 1 if purpose_slug and purpose_slug in usages else 0

        # --- feature gốc ---
        row = {
            "abs_price_diff": abs(price - budget),
            "brand_match": 1 if pref_brand and pref_brand == lap_brand else 0,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "price": price,
            "usage_study":   1 if "hoc_tap"   in usages else 0,
            "usage_design":  1 if "do_hoa"    in usages else 0,
            "usage_gaming":  1 if "gaming"    in usages else 0,
            "usage_office":  1 if "van_phong" in usages else 0,
            "usage_work":    1 if ("lap_trinh" in usages or "doanh_nhan" in usages) else 0,
            "has_dgpu": 1 if has_discrete_gpu(it.get("gpu", "")) else 0,
            "purpose_match": purpose_match,
        }

        # --- ONE-HOT CLUSTER (cluster_0..cluster_{K-1}) ---
        # cluster đã được gắn vào laptop từ load_laptops_with_stock()
        cid = it.get("kmeans_cluster")
        try:
            cid_int = int(cid) if cid is not None else -1
        except Exception:
            cid_int = -1

        for k in range(KMEANS_N_CLUSTERS):
            col_name = f"cluster_{k}"
            row[col_name] = 1 if cid_int == k else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # đảm bảo có đủ tất cả FEATURE_COLS (kể cả cluster_*), thiếu thì fill 0
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
    if "học" in t or "hoc" in t or "sinh viên" in t or "sv" in t:
        usages.append("hoc_tap")

    if ("văn phòng" in t or "van phong" in t or "office" in t or
        "word" in t or "excel" in t or "powerpoint" in t):
        usages.append("van_phong")

    if ("dev" in t or "lập trình" in t or "lap trinh" in t or
        "program" in t or "code" in t):
        usages.append("lap_trinh")

    if "game" in t or "chơi game" in t:
        usages.append("gaming")

    if "thiết kế" in t or "design" in t or "đồ hoạ" in t or "do hoa" in t:
        usages.append("do_hoa")

    if "doanh nhân" in t or "kinh doanh" in t or "business" in t:
        usages.append("doanh_nhan")

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
    print("MOBILE PING >>>", payload) 
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
        .limit(10)
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
        elif p < 20_000_000:
            return "10-20"
        elif p < 30_000_000:
            return "20-30"
        elif p < 40_000_000:
            return "30-40"
        else:
            return ">40"

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

def get_clusters_for_usage(usages):
    """
    Map usage (hoc_tap, gaming, do_hoa, van_phong, doanh_nhan, ...) 
    sang danh sách cluster_id tương ứng (theo tag trong KMEANS_CLUSTER_CONFIG).
    """
    if not usages:
        return None
    if not KMEANS_CLUSTER_CONFIG or "clusters" not in KMEANS_CLUSTER_CONFIG:
        return None

    if isinstance(usages, str):
        usages = [usages]
    usages = [str(u).lower() for u in usages]

    clusters = []
    for cid_str, info in KMEANS_CLUSTER_CONFIG.get("clusters", {}).items():
        tag = (info.get("tag") or "").lower()
        if not tag:
            continue
        # nếu tag của cụm trùng với bất kì usage nào → cho cụm đó
        if tag in usages:
            try:
                clusters.append(int(cid_str))
            except Exception:
                pass

    # nếu không map được cụm nào → trả None để không lọc
    return clusters or None

def cf_recommend_scores_for_user(user_id: str, topn: int = 200) -> dict:
    """
    Trả về dict: { laptop_id (UUID str) : score_cf }
    Dùng ALS implicit model.
    """
    if not user_id or CF_MODEL is None or CF_USER_ITEM_MATRIX is None:
        return {}

    uid = CF_USER2IDX.get(user_id)
    if uid is None:
        # user chưa từng tương tác => cold-start
        return {}

    # implicit ALS làm việc với item_user = matrix.T
    # recommend(user, user_items, N)
    # user_items ở đây là hàng tương ứng của user trong user_item_matrix
    try:
        user_items = CF_USER_ITEM_MATRIX[uid]  # row dạng 1 x num_items (csr)
    except Exception:
        return {}

    # ALS trong implicit expects item_user (items x users), nhưng recommend()
    # trong version mới cho phép truyền user_items là sparse row vector
    recommended = CF_MODEL.recommend(
        userid=uid,
        user_items=user_items,
        N=topn,
        filter_already_liked_items=False,
    )

    # recommended: list (item_idx, score)
    scores = {}
    for item_idx, score in recommended:
        if 0 <= item_idx < len(CF_IDX2ITEM):
            lap_id = str(CF_IDX2ITEM[item_idx])
            scores[lap_id] = float(score)

    return scores


def price_bucket_of_item(item):
    p = item.get("price") or 0
    if p < 10_000_000:
        return "<10"
    elif p < 20_000_000:
        return "10-20"
    elif p < 30_000_000:
        return "20-30"
    elif p < 40_000_000:
        return "30-40"
    else:
        return ">40"
    
def discount_percent(lap: dict) -> float:
    """
    Tính % giảm giá cho 1 laptop.
    Nếu không có promo_price hoặc price <= 0 thì trả về 0.0
    """
    try:
        price = float(lap.get("price") or 0)
        promo = float(lap.get("promo_price") or 0)
    except Exception:
        return 0.0

    if price <= 0 or promo <= 0 or promo >= price:
        return 0.0

    return (price - promo) / price

def apply_personalization(laptops: list, user_id: str):
    """
    Điều chỉnh _score dựa trên gu riêng của user:
    - brand: hãng user hay click
    - price_bucket: tầm giá user hay xem
    - ram_gb: RAM user hay chọn
    """
    if not user_id:
        return laptops

    brand_scores, price_bucket_scores, ram_scores = get_user_preference(user_id)

    # nếu user chưa có lịch sử gì thì thôi, trả y nguyên
    if not (brand_scores or price_bucket_scores or ram_scores):
        return laptops

    max_brand = max(brand_scores.values()) if brand_scores else 1
    max_bucket = max(price_bucket_scores.values()) if price_bucket_scores else 1
    max_ram = max(ram_scores.values()) if ram_scores else 1

    for lap in laptops:
        base_score = float(lap.get("_score", 0.0))
        bonus = 0.0

        # --- BRAND ---
        b = (lap.get("brand") or "").lower()
        if b in brand_scores and max_brand > 0:
            # user càng click brand đó nhiều, bonus càng lớn
            bonus += 0.08 * (brand_scores[b] / max_brand)

        # --- PRICE BUCKET ---
        pb = price_bucket_of_item(lap)
        if pb in price_bucket_scores and max_bucket > 0:
            bonus += 0.05 * (price_bucket_scores[pb] / max_bucket)

        # --- RAM ---
        ram = lap.get("ram_gb")
        if ram in ram_scores and max_ram > 0:
            bonus += 0.03 * (ram_scores[ram] / max_ram)

        lap["_score"] = base_score + bonus

    # sắp xếp lại sau khi cộng bonus
    laptops.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return laptops


# ========== SERVE REACT ==========
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(app.static_folder, "static"), path)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")


# ========== API GỢI Ý LAPTOP ==========
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
    return False


def apply_explicit_filters(laptops: list, filters: dict) -> list:
    if not filters:
        return laptops

    price_f   = filters.get("price") or {}
    brand_f   = filters.get("brand")
    ram_f     = filters.get("ram")
    gpu_f     = filters.get("gpu")
    purpose_f = filters.get("purpose") or filters.get("usage")  # phòng khi dùng key 'usage'

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

        # purpose (mục đích / usage)
        if ok and purpose_f:
            lp = (lap.get("purpose") or "").lower()
            if isinstance(purpose_f, list):
                allow = [p.lower() for p in purpose_f]
                if lp not in allow:
                    ok = False
            else:
                if lp != str(purpose_f).lower():
                    ok = False

        if ok:
            out.append(lap)

    return out



def load_laptops_with_stock():
    """
    Helper: lấy tất cả laptops + tồn kho từ view laptop_stock,
    gắn thêm:
      - stock_qty: tồn kho hiện tại
      - in_stock: còn hàng hay không
      - sold_count: tổng số lượng đã bán (từ các đơn 'Hoàn thành')
    Dùng nội bộ, KHÔNG gắn route.
    """
    # 1. Lấy toàn bộ laptop
    lap_res = supabase.table("laptops").select("*").execute()
    laptops = lap_res.data or []

    # 2. Lấy tồn kho từ view laptop_stock (id, stock_qty)
    stock_res = supabase.table("laptop_stock").select("id,stock_qty").execute()
    stock_rows = stock_res.data or []

    stock_map = {row["id"]: row.get("stock_qty") or 0 for row in stock_rows}

    # 3. Tính tổng đã bán (sold_count) dựa trên order_items + orders 'Hoàn thành'
    sold_map = Counter()
    try:
        # lấy tất cả đơn Hoàn thành
        ord_res = (
            supabase_admin.table("orders")
            .select("id,status")
            .eq("status", "Hoàn thành")
            .execute()
        )
        ord_rows = ord_res.data or []
        order_ids = [o["id"] for o in ord_rows]

        if order_ids:
            # lấy các dòng order_items của những đơn đó
            oi_res = (
                supabase_admin.table("order_items")
                .select("order_id,laptop_id,quantity")
                .in_("order_id", order_ids)
                .execute()
            )
            oi_rows = oi_res.data or []

            for r in oi_rows:
                lid = r.get("laptop_id")
                if not lid:
                    continue
                qty = int(r.get("quantity") or 0)
                if qty > 0:
                    sold_map[lid] += qty
    except Exception as e:
        print("load_laptops_with_stock sold_count error:", e)

    # 4. Gắn stock_qty + in_stock + sold_count vào từng laptop
    for lap in laptops:
        lap_id = lap.get("id")

        qty = stock_map.get(lap_id, 0)
        lap["stock_qty"] = qty
        lap["in_stock"] = qty > 0

        lap["sold_count"] = int(sold_map.get(lap_id, 0))
        try:
            lap_cluster = LAPTOP_CLUSTER_MAP.get(str(lap_id))
        except Exception:
            lap_cluster = None
        lap["kmeans_cluster"] = lap_cluster
    return laptops



@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    t0 = time.perf_counter()
    body = request.get_json(force=True) or {}

    # --- Thông tin chung từ body ---
    device = body.get("device") or "android"
    user_id = body.get("user_id")

    # Android gửi "query", dashboard có thể gửi "raw_query"
    raw_query = (body.get("raw_query") or body.get("query") or "").strip() or None

    # Lọc giá
    min_price = body.get("min_price")
    max_price = body.get("max_price")

    # Brand & purpose từ body
    brand_from_body = body.get("brand")
    purpose_from_body = body.get("purpose")  # từ app (hoc_tap, gaming,...)

    # Thông tin cho đánh giá mô hình (app gửi lên)
    raw_qt = (body.get("query_type") or "").strip().lower()
    source_model = body.get("source_model") or "content_based_v1"

    topk = int(body.get("topk") or 10)

    # NEW: tab đặc biệt cho mobile: "sale" / "hot" / "best_seller"
    special = (body.get("special") or "").strip().lower() or None

    # --- Phân tích nội dung câu hỏi ---
    struct = parse_user_query_to_struct(raw_query or "")
    # struct có thể có: brand, usage, budget, needs_dgpu, avoid_dgpu,...

    # ==== HẬU XỬ LÝ GIÁ TỪ CÂU HỎI ====
    budget = struct.get("budget")
    text = (raw_query or "").lower()

    # Chỉ map khi body chưa gửi min/max_price
    if budget is not None and min_price is None and max_price is None:
        # Nếu câu có "dưới / nhỏ hơn / tối đa / <" → xem budget là GIÁ TRẦN (max_price)
        if any(kw in text for kw in ["dưới", "<", "nhỏ hơn", "tối đa", "max"]):
            max_price = budget
        # Nếu câu có "trên / lớn hơn / tối thiểu / >" → xem budget là GIÁ SÀN (min_price)
        elif any(kw in text for kw in ["trên", ">", "lớn hơn", "tối thiểu", "min"]):
            min_price = budget
        else:
            # Câu kiểu "khoảng 20tr", "tầm 15tr" → cho là trần trên cho dễ dùng
            max_price = budget

    # struct có thể có: brand, usage, budget, needs_dgpu, avoid_dgpu,...

    # Ưu tiên brand trong câu, nếu không có thì dùng brand filter
    if struct.get("brand"):
        brand = struct["brand"]
    else:
        brand = (brand_from_body or "").lower() or None

    # Hợp nhất "mục đích dùng" từ câu & từ filter
    usage = struct.get("usage") or purpose_from_body or None

    # --- Load danh sách laptop + tồn kho ---
    laptops = load_laptops_with_stock()   # gắn stock_qty + in_stock
    laptops = [lap for lap in laptops if lap.get("in_stock")]  # chỉ recommend hàng còn

    # --- Apply các filter rõ ràng (giá, brand, gpu, purpose) ---
    filters = {}

    if min_price is not None or max_price is not None:
        filters["price"] = {
            "min": min_price,
            "max": max_price,
        }

    if brand:
        filters["brand"] = brand

    # Dùng usage/purpose để lọc theo laptops.purpose
    if usage:
        filters["purpose"] = usage

    # GPU theo nội dung câu
    if struct.get("needs_dgpu"):
        filters["gpu"] = "discrete"
    elif struct.get("avoid_dgpu"):
        filters["gpu"] = "integrated"

    laptops = apply_explicit_filters(laptops, filters)

        # --- LỌC THEO CỤM K-MEANS DỰA TRÊN USAGE (NẾU CÓ) ---
    # usage hiện tại lấy từ struct hoặc từ body / purpose_from_body
    usage_list = struct.get("usage") or usage
    target_clusters = get_clusters_for_usage(usage_list)

    if target_clusters is not None:
        filtered_by_cluster = []
        for lap in laptops:
            cid = lap.get("kmeans_cluster")
            try:
                cid_int = int(cid) if cid is not None else None
            except Exception:
                cid_int = None

            if cid_int in target_clusters:
                filtered_by_cluster.append(lap)

        # fallback: nếu lọc theo cụm ra rỗng thì giữ nguyên danh sách cũ
        if filtered_by_cluster:
            laptops = filtered_by_cluster

    # ===== PREPARE query_type (chỉ còn 4 loại chính) =====
    valid_types = {"keyword", "filter_only", "hybrid", "content_rec", "browse_all"}

    def infer_query_type(raw_q, has_filters_flag):
        """
        Suy luận kiểu truy vấn dựa theo hành vi thực tế:
        - Có query, không filter  -> keyword
        - Không query, có filter  -> filter_only
        - Có cả 2                 -> hybrid
        - Không query, không filter (nhấn Gợi ý ML, đề xuất mặc định...) -> content_rec
        """
        if raw_q and has_filters_flag:
            return "hybrid"
        if raw_q and not has_filters_flag:
            return "keyword"
        if (not raw_q) and has_filters_flag:
            return "filter_only"
        return "content_rec"

    has_filters_flag = has_explicit_filters(filters)

    if raw_qt in valid_types:
        # nếu app gửi đúng thì dùng luôn
        query_type = raw_qt
    else:
        # nếu app gửi bậy / null -> tự suy luận
        query_type = infer_query_type(raw_query, has_filters_flag)

    # --- Nếu không còn máy nào sau khi lọc ---
    if not laptops:
        result_laptops = []
        result_ids = []

    else:
        # ====== 3 TAB ĐẶC BIỆT: SALE / HOT / BEST_SELLER ======
        if special in ("sale", "hot", "best_seller"):
            # 3 mode này ưu tiên rule đơn giản thay vì ML
            if special == "sale":
                # SALE KHỦNG: giảm giá >= 12%
                sale_list = []
                for lap in laptops:
                    disc = discount_percent(lap)  # 👈 dùng helper
                    if disc >= 0.12:             # "sale khủng" >= 12%
                        lap["_score"] = disc
                        sale_list.append(lap)

                # fallback: nếu chẳng có máy nào ≥12%, cho phép >0%
                if not sale_list:
                    for lap in laptops:
                        disc = discount_percent(lap)
                        if disc > 0:
                            lap["_score"] = disc
                            sale_list.append(lap)

                # Nếu vẫn rỗng luôn thì dùng lại list gốc
                if sale_list:
                    sale_list.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
                    result_laptops = sale_list[:topk]
                else:
                    result_laptops = laptops[:topk]
            
            elif special == "hot":
                # 1. Lấy mốc 30 ngày gần nhất
                start_30d = datetime.now(timezone.utc) - timedelta(days=30)
                start_iso = start_30d.isoformat()

                # 2. Dùng supabase_admin để không bị RLS chặn
                click_res = (
                    supabase_admin
                    .table("laptop_click_logs")
                    .select("laptop_id, created_at")
                    .gte("created_at", start_iso)
                    .execute()
                )
                click_rows = click_res.data or []

                click_counter = Counter()
                for r in click_rows:
                    lid = r.get("laptop_id")
                    if lid:
                        click_counter[lid] += 1

                for lap in laptops:
                    lid = lap.get("id")
                    lap["_score"] = float(click_counter.get(lid, 0) or 0)
        # nếu trong load_laptops_with_stock có created_at, ta có luôn lap["created_at"]

    # nếu trong dict có created_at (ISO string) thì parse sang số cho dễ sort
                def created_ts(lap):
                    s = lap.get("created_at")
                    if not s:
                        return 0.0
                    try:
                        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        return 0.0

                laptops.sort(
                    key=lambda x: (x.get("_score", 0.0), created_ts(x), x.get("id") or ""),
                    reverse=True,
                )

                result_laptops = laptops[:topk]



            elif special == "best_seller":
            # BÁN CHẠY: dựa trên số lượng đã bán trong order_items,
            # nhưng chỉ tính các đơn "Hoàn thành" trong 30 ngày gần nhất
                sales_counter = Counter()

                try:
                # 1️⃣ Lấy tất cả orders "Hoàn thành"
                    ord_res = (
                        supabase_admin.table("orders")
                        .select("id,status,created_at")
                        .eq("status", "Hoàn thành")
                        .execute()
                    )
                    ord_rows = ord_res.data or []

                # 2️⃣ Tính mốc 30 ngày trước
                    since = datetime.now(timezone.utc) - timedelta(days=30)

                # 3️⃣ Lọc ra các order_id trong 30 ngày gần nhất (lọc bằng Python)
                    ok_ids = []
                    for o in ord_rows:
                        created_str = o.get("created_at")
                        if not created_str:
                            continue
                        try:
                        # Supabase trả ISO string, ví dụ "2025-11-27T12:34:56.123456+00:00"
                            created_dt = datetime.fromisoformat(
                                created_str.replace("Z", "+00:00")
                            )
                        except Exception:
                            continue

                        if created_dt >= since:
                            ok_ids.append(o["id"])

                # 4️⃣ Nếu có đơn hợp lệ thì truy vấn order_items
                    if ok_ids:
                        item_res = (
                            supabase_admin.table("order_items")
                            .select("order_id,laptop_id,quantity")
                            .in_("order_id", ok_ids)
                            .execute()
                        )
                        item_rows = item_res.data or []
                        for r in item_rows:
                            lid = r.get("laptop_id")
                            if not lid:
                                continue
                            qty = int(r.get("quantity") or 1)
                            sales_counter[lid] += qty
        
                except Exception as e:
                    print("BEST_SELLER_QUERY_ERROR:", e)

            # 5️⃣ Gán score cho từng laptop theo số lượng đã bán
                for lap in laptops:
                    lid = lap.get("id")
                    lap["_score"] = float(sales_counter.get(lid, 0))

            # 6️⃣ Sắp xếp và lấy topk
                laptops.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
                result_laptops = laptops[:topk]



            # kết quả chung cho 3 tab đặc biệt
            result_ids = [to_uuid_or_none(l.get("id")) for l in result_laptops]

        else:
    # ====== ML RANKING / FALLBACK ======
            if ML_MODEL is not None and FEATURE_COLS:
                feats = build_features_for_items(struct, laptops)

                if hasattr(ML_MODEL, "predict_proba"):
                    scores = ML_MODEL.predict_proba(feats)[:, 1]
                else:
                    scores = ML_MODEL.predict(feats)

                for lap, s in zip(laptops, scores):
            # ContentScore ban đầu
                    lap["_score"] = float(s)

                laptops.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
            else:
        # --- Fallback: sort theo "gần ngân sách" ---
                budget = struct.get("budget")
                if budget:
                    for lap in laptops:
                        p = float(lap.get("price") or 0)
                        lap["_score"] = -abs(p - budget)
                    laptops.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
                else:
            # nếu không có budget & không có ML, đảm bảo vẫn có _score
                    for lap in laptops:
                        lap["_score"] = 0.0

    # ⭐⭐ HYBRID: CONTENT + CF (ALS) ⭐⭐
    # Nếu user có đủ tương tác trong CF thì trộn;
    # nếu không, hàm sẽ trả về nguyên danh sách (chỉ content-based).
            if user_id:
                laptops = apply_hybrid_scores(laptops, user_id, alpha=0.7)

    # ⭐⭐ CÁ NHÂN HÓA THEO USER (rule-based) ⭐⭐
            if user_id:
                laptops = apply_personalization(laptops, user_id)

    # cắt topk sau khi đã hybrid + cá nhân hoá
            result_laptops = laptops[:topk]
            result_ids = [to_uuid_or_none(l.get("id")) for l in result_laptops]


           

    # ⭐ LUÔN luôn tính latency_ms sau khi xử lý xong, ngoài if/else
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # --- Chuẩn bị struct để log ---
    parsed_struct = {
        "brand": brand,
        "usage": usage,
        "budget": struct.get("budget"),
        "min_price": min_price,
        "max_price": max_price,
        "topk": topk,
    }

    # Chỉ log khi có gì đó "đáng log"
    should_log = bool(
        (raw_query and raw_query.strip())
        or (min_price is not None)
        or (max_price is not None)
        or brand
        or usage
        or query_type == "content_rec"   # 👈 thêm dòng này
    )

    if should_log:
        log_search(
            raw_query=raw_query,
            parsed_struct=parsed_struct,
            result_ids=[rid for rid in result_ids if rid],
            device=device,
            user_id=user_id,
            brand=brand,
            usage=usage,
            budget=struct.get("budget"),
            min_price=min_price,
            max_price=max_price,
            query_type=query_type,      # 🔹 lấy từ body (keyword/filter_only/hybrid/content_rec)
            topk=topk,
            latency_ms=latency_ms,      # 🔹 giờ đã chắc chắn được gán
            source_model=source_model,  # 🔹 lấy từ body (baseline/content_based/hybrid...)
        )

    return jsonify({
        "ok": True,
        "items": result_laptops,
        "latency_ms": latency_ms,
    })








# ========== WEB ADMIN ==========
@app.route("/admin")
def admin_home():
    return render_template("admin.html")

@app.route("/admin/orders")
def admin_orders_page():
   
    return render_template("admin_orders.html", order_statuses=ORDER_STATUSES)


def fetch_all_search_logs():
    """
    Lấy toàn bộ search_logs (vượt giới hạn 1000 dòng của Supabase
    bằng cách phân trang).
    """
    all_logs = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase.table("search_logs")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = res.data or []
        all_logs.extend(rows)

        # nếu nhận < page_size bản ghi thì coi như hết dữ liệu
        if len(rows) < page_size:
            break

        offset += page_size

    return all_logs


@app.route("/admin/stats")
def admin_stats():
    logs = fetch_all_search_logs()
    return render_template("admin_stats.html", logs=logs)




@app.route("/admin/api/stats_json")
def stats_json():
    # ==== Lấy log thô (tất cả) ====
    logs = fetch_all_search_logs()


    # ==== Khởi tạo thống kê ====
    price_query_buckets = {
        "<10tr": 0,
        "10-20tr": 0,
        "20-30tr": 0,
        "30-40tr": 0,
        ">40tr": 0,
        "Tất cả": 0,
    }

    usage_query_counts = {
        "hoc_tap": 0,
        "van_phong": 0,
        "do_hoa": 0,
        "lap_trinh": 0,
        "gaming": 0,
        "doanh_nhan": 0,
    }

    KNOWN_BRANDS = ["apple", "dell", "hp", "lenovo", "asus", "acer", "msi"]
    brand_counts = {b: 0 for b in KNOWN_BRANDS}

    # ==== Top user dùng gợi ý ====
    counts_by_user = {}
    for row in logs:
        uid = row.get("user_id")
        if not uid:
            continue
        counts_by_user[uid] = counts_by_user.get(uid, 0) + 1

    top_user_ids = sorted(
        counts_by_user.items(), key=lambda x: x[1], reverse=True
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

    # ==== Quét từng log để thống kê brand / giá / usage ====
    for row in logs:
        parsed = row.get("parsed_struct") or {}
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = {}

        # ----- BRAND -----
        b = (parsed.get("brand") or row.get("brand") or "").lower()
        if b in brand_counts:
            brand_counts[b] += 1

        # ----- PRICE BUCKET -----
        minp = parsed.get("min_price")
        maxp = parsed.get("max_price")

        # fallback sang cột thường nếu JSON không có
        if minp is None and maxp is None:
            minp = row.get("min_price")
            maxp = row.get("max_price")

        budget = parsed.get("budget")
        if budget is None:
            budget = row.get("budget")

        if minp is None and maxp is None and budget is None:
            # hoàn toàn không có thông tin giá
            price_query_buckets["Tất cả"] += 1
        else:
            # ưu tiên max_price, rồi min_price, cuối cùng budget
            p = maxp if maxp is not None else (minp if minp is not None else budget)

            try:
                v = float(p)
            except Exception:
                price_query_buckets["Tất cả"] += 1
            else:
                if v < 10_000_000:
                    price_query_buckets["<10tr"] += 1
                elif v < 20_000_000:
                    price_query_buckets["10-20tr"] += 1
                elif v < 30_000_000:
                    price_query_buckets["20-30tr"] += 1
                elif v < 40_000_000:
                    price_query_buckets["30-40tr"] += 1
                else:
                    price_query_buckets[">40tr"] += 1

        # ----- USAGE (mục đích) -----
        u = parsed.get("usage")
        if not u:
            u = row.get("usage")  # 👈 dùng thêm cột usage em vừa fake

        if isinstance(u, list):
            usages = u
        elif isinstance(u, str) and u.strip():
            usages = [u]
        else:
            usages = []

        for x in usages:
            key = str(x).lower().strip()

            # 1) Trường hợp em đã lưu slug luôn (van_phong, gaming...)
            if key in usage_query_counts:
                usage_query_counts[key] += 1
                continue

            # 2) Trường hợp là tiếng Anh / tiếng Việt tự do → map sang slug
            mapped = None
            if key in ("study", "hoc", "học", "sinh viên", "sinh vien"):
                mapped = "hoc_tap"
            elif key in ("office", "van phong", "văn phòng"):
                mapped = "van_phong"
            elif key in ("design", "do hoa", "đồ hoạ", "đồ họa"):
                mapped = "do_hoa"
            elif key in ("dev", "work", "lap trinh", "lập trình", "programmer", "code"):
                mapped = "lap_trinh"
            elif key in ("gaming", "game", "chơi game"):
                mapped = "gaming"
            elif key in ("doanh nhân", "doanh nhan", "business"):
                mapped = "doanh_nhan"

            if mapped and mapped in usage_query_counts:
                usage_query_counts[mapped] += 1

    # ==== Brand list cho chart ====
    brand_from_logs = [
        {"brand": b, "total": brand_counts[b]} for b in KNOWN_BRANDS
    ]

    # ==== Click logs ====
        # ==== Click logs: TOP 10 laptop được click nhiều nhất 30 ngày gần nhất ====
    start_30d = datetime.now(timezone.utc) - timedelta(days=30)
    start_iso = start_30d.isoformat()

    click_res = (
        supabase.table("laptop_click_logs")
        .select("laptop_id, created_at")
        .gte("created_at", start_iso)
        .execute()
    )
    click_rows = click_res.data or []

    click_counter = Counter()
    for r in click_rows:
        lid = r.get("laptop_id")
        if lid:
            click_counter[lid] += 1

    # 2) Top 10 laptop theo số click
    top_click_ids = sorted(
        click_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # 3) Lấy thông tin laptop
    lap_map = {}
    if top_click_ids:
        lid_list = [lid for lid, _ in top_click_ids]

        # lấy name, image
        laps_res = (
            supabase.table("laptops")
            .select("id, name, image_url")
            .in_("id", lid_list)
            .execute()
        )
        laps = laps_res.data or []
        lap_map = {lap["id"]: lap for lap in laps}

        # 🔥 LẤY THÊM TỒN KHO TỪ VIEW laptop_stock
        stock_res = (
            supabase.table("laptop_stock")
            .select("id, stock_qty")      # đúng tên cột của view
            .in_("id", lid_list)          # join theo id
            .execute()
        )

        stock_map = {}
        for row in (stock_res.data or []):
            # map: id (uuid) -> số lượng tồn
            stock_map[row["id"]] = row.get("stock_qty") or 0
    else:
        stock_map = {}

    # 4) Gộp lại thành danh sách trả ra cho dashboard
    top_clicked = []
    for lid, total in top_click_ids:
        lap = lap_map.get(lid, {})
        qty = stock_map.get(lid, 0)
        in_stock = qty > 0

        top_clicked.append({
            "laptop_id": lid,
            "name": lap.get("name") or f"#{str(lid)[:6]}",
            "image_url": lap.get("image_url"),
            "total_click": total,
            "stock_qty": int(qty),
            "in_stock": in_stock,
        })

   


    # ==== Cart top laptops ====
    traffic_logs = load_traffic()

    cart_res = (
        supabase.table("carts")
        .select("laptop_id, quantity")
        .execute()
    )
    cart_rows = cart_res.data or []

   
    cart_counter = Counter()
    for row in cart_rows:
        lid = row.get("laptop_id")
        if not lid:
            continue
        qty = row.get("quantity") or 1
        cart_counter[lid] += qty

    top_cart_laptops = []
    for lid, total in cart_counter.most_common(10):
        lap_res = (
            supabase.table("laptops")
            .select("name,image_url,price")
            .eq("id", lid)
            .limit(1)
            .execute()
        )
        laps = lap_res.data or []
        lap = laps[0] if laps else {}
        top_cart_laptops.append({
            "laptop_id": lid,
            "name": lap.get("name") or lid,
            "image_url": lap.get("image_url"),
            "price": lap.get("price"),
            "total_cart": total,
        })

    def parse_ts(s):
        if not s:
            return None
        if isinstance(s, datetime):
            return s
        try:
            # '2025-11-25T05:12:34+00:00' hoặc '2025-11-25T05:12:34Z'
            s2 = str(s).replace("Z", "+00:00")
            return datetime.fromisoformat(s2)
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    today = now.date()

# ===== ĐẦU THÁNG NÀY / THÁNG TRƯỚC =====
# đầu tháng này
    start_this_month = today.replace(day=1)

# đầu tháng trước
    if start_this_month.month == 1:
        start_prev_month = date(start_this_month.year - 1, 12, 1)
    else:
        start_prev_month = date(
            start_this_month.year,
            start_this_month.month - 1,
            1
        )

# helper: tháng của 1 ngày bất kỳ → đầu tháng kế tiếp
    def next_month(d: date) -> date:
        if d.month == 12:
            return date(d.year + 1, 1, 1)
        return date(d.year, d.month + 1, 1)

    end_this_month = next_month(start_this_month)
    end_prev_month = start_this_month  # vì đây là đầu tháng này

# ===== 7 & 30 NGÀY GẦN NHẤT =====
# 7 ngày gần nhất (tính cả hôm nay)
    seven_days_ago = today - timedelta(days=6)

# 30 ngày gần nhất (tính cả hôm nay)
    thirty_days_ago = today - timedelta(days=29)

# Lấy tất cả orders
    orders_res = (
        supabase.table("orders")
        .select("id,total_amount,status,created_at,user_id")
        .execute()
    )
    orders = orders_res.data or []

    revenue_today = 0.0
    revenue_7days = 0.0
    revenue_this_month = 0.0
    revenue_prev_month = 0.0

    orders_today = 0
    orders_pending = 0
    orders_success = 0
    orders_cancelled = 0

    shipping_in_transit = 0
    shipping_delivered = 0
    shipping_cancelled = 0
    shipping_this_month = {
    "wait_confirm": 0,   # 👈 Chờ xác nhận (tháng này)
    "wait_pickup": 0,    # 👈 Chờ lấy hàng (tháng này)
    "in_transit": 0,     # Đang giao / chờ giao
    "delivered": 0,      # Hoàn thành
    "cancelled": 0,      # Đã hủy
    "tra_hang": 0,       # Trả hàng
}


    orders_wait_confirm = 0   # Chờ xác nhận
    orders_wait_pickup = 0    # Chờ lấy hàng
    orders_shipping = 0       # Chờ giao hàng / đang giao
    orders_return = 0         # 👈 Đơn trả hàng
    shipping_return = 0
    buyers_last_30d = set()

    pending_statuses = {"Chờ xác nhận", "Chờ lấy hàng", "Chờ giao hàng"}    # muốn thì thêm "Chờ giao hàng"
    success_statuses = {"Hoàn thành", "Đã giao"}
    cancelled_statuses = {"Đã hủy", "Đã huỷ"}

# ------------- VÒNG LẶP CHÍNH ------------
    for o in orders:
        dt = parse_ts(o.get("created_at"))
        if not dt:
            continue
        d = dt.date()
        amt = float(o.get("total_amount") or 0)
        st = (o.get("status") or "").strip()

        in_this_month = start_this_month <= d < end_this_month

    # ---- Chờ xác nhận ----
        if st == "Chờ xác nhận":
            orders_wait_confirm += 1
            orders_pending += 1
            if in_this_month:
                shipping_this_month["wait_confirm"] += 1

    # ---- Chờ lấy hàng ----
        elif st == "Chờ lấy hàng":
            orders_wait_pickup += 1
            orders_pending += 1
            if in_this_month:
                shipping_this_month["wait_pickup"] += 1

    # ---- Chờ giao / đang giao ----
        elif st == "Chờ giao hàng":
            orders_shipping += 1
            orders_pending += 1
            shipping_in_transit += 1
            if in_this_month:
                shipping_this_month["in_transit"] += 1

    # ---- Trả hàng ----
        elif st == "Trả hàng":
            orders_return += 1
            shipping_return += 1
            if in_this_month:
                shipping_this_month["tra_hang"] += 1

    # ---- Thành công ----
        elif st in success_statuses:
            orders_success += 1
            shipping_delivered += 1
            if in_this_month:
                shipping_this_month["delivered"] += 1

    # ---- Hủy ----
        elif st in cancelled_statuses:
            orders_cancelled += 1
            shipping_cancelled += 1
            if in_this_month:
                shipping_this_month["cancelled"] += 1

    # ==== Doanh thu CHỈ tính cho đơn thành công ====
        if st in success_statuses:
            if d == today:
                revenue_today += amt
                orders_today += 1

            if d >= seven_days_ago:
                revenue_7days += amt

            if start_this_month <= d < end_this_month:
                revenue_this_month += amt

            if start_prev_month <= d < end_prev_month:
                revenue_prev_month += amt


    # ---- Conversion 30 ngày (người mua) ----
        if d >= thirty_days_ago:
            uid = o.get("user_id")
            if uid:
                buyers_last_30d.add(uid)


        # ==== TOP LAPTOP BÁN CHẠY 30 NGÀY GẦN NHẤT ====
    sold_counter = Counter()
    ok_ids = []

    # Lọc các đơn hoàn thành trong 30 ngày gần nhất
    for o in orders:
        dt = parse_ts(o.get("created_at"))
        if not dt:
            continue
        d = dt.date()
        st = (o.get("status") or "").strip()
        if st in success_statuses and d >= thirty_days_ago:
            ok_ids.append(o["id"])

    top_sold_laptops_30d = []

    if ok_ids:
        # Lấy order_items của các đơn đó
        oi_res = (
            supabase.table("order_items")
            .select("order_id,laptop_id,quantity")
            .in_("order_id", ok_ids)
            .execute()
        )
        oi_rows = oi_res.data or []

        for r in oi_rows:
            lid = r.get("laptop_id")
            if not lid:
                continue
            qty = int(r.get("quantity") or 1)
            sold_counter[lid] += qty

        if sold_counter:
            # Lấy top 10 laptop bán nhiều nhất
            top_ids = [lid for lid, _ in sold_counter.most_common(10)]

            laps_res2 = (
                supabase.table("laptops")
                .select("id,name,image_url")
                .in_("id", top_ids)
                .execute()
            )
            laps2 = laps_res2.data or []
            lap_map2 = {lap["id"]: lap for lap in laps2}

            for lid, total in sold_counter.most_common(10):
                lap = lap_map2.get(lid, {})
                top_sold_laptops_30d.append({
                    "laptop_id": lid,
                    "name": lap.get("name") or f"#{str(lid)[:6]}",
                    "image_url": lap.get("image_url"),
                    "total_sold_30d": int(total),
                })

    # --- GỘP DOANH THU THEO THÁNG ---
    monthly_revenue = defaultdict(float)

    for o in orders:
        dt = parse_ts(o.get("created_at"))
        if not dt:
            continue
        d = dt.date()
        st = (o.get("status") or "").strip()
        amt = float(o.get("total_amount") or 0)

        if st in success_statuses:
            key = (d.year, d.month)
            monthly_revenue[key] += amt

    # Lấy 5 tháng gần nhất
    last_5 = []
    for (y, m), total in sorted(monthly_revenue.items())[-5:]:
        label = f"{m:02d}/{y}"       # ví dụ "11/2025"
        last_5.append({
            "label": label,
            "total": total,
        })
# % tăng/giảm doanh thu tháng này so với tháng trước
    if revenue_prev_month > 1e-6:
        month_change_percent = (
            (revenue_this_month - revenue_prev_month) / revenue_prev_month * 100.0
        )
    else:
        month_change_percent = None  # tránh chia 0

# conversion: người xem → người mua (30 ngày)
    clicks_30d_res = (
        supabase.table("laptop_click_logs")
        .select("user_id,created_at")
        .execute()
    )
    click_30d_rows = clicks_30d_res.data or []

    viewers_last_30d = set()
    for r in click_30d_rows:
        dt = parse_ts(r.get("created_at"))
        if not dt:
            continue
        if dt.date() >= thirty_days_ago:
            uid = r.get("user_id")
            if uid:
                viewers_last_30d.add(uid)

    num_viewers = len(viewers_last_30d)
    num_buyers = len(buyers_last_30d)
    if num_viewers > 0:
        conversion_rate = num_buyers / num_viewers
    else:
        conversion_rate = 0.0


    # =================================================================
    #                           RESPONSE JSON
    # =================================================================
    return jsonify({
        "logs": logs,
        "top_search_users": top_users_from_logs,
        "brand_from_logs": brand_from_logs,
        "top_clicked_laptops": top_clicked,
        "traffic_logs": traffic_logs,
        "top_cart_laptops": top_cart_laptops,
        "price_query_buckets": price_query_buckets,
        "usage_query_counts": usage_query_counts,
        "top_sold_laptops_30d": top_sold_laptops_30d,

        # ---- Phần mới cho dashboard cửa hàng ----
        "revenue": {
            "today": revenue_today,
            "last7_days": revenue_7days,
            "this_month": revenue_this_month,
            "prev_month": revenue_prev_month,
            "month_change_percent": month_change_percent,
            "last_5_months": last_5,
        },
        "orders_summary": {
            "today": orders_today,
            "pending_wait_confirm": orders_wait_confirm,
            "pending_wait_pickup": orders_wait_pickup,
            "pending_shipping": orders_shipping,
            "success": orders_success,
            "cancelled": orders_cancelled,
            "tra_hang": orders_return,
        },
        "shipping_status": {
            "in_transit": shipping_in_transit,
            "delivered": shipping_delivered,
            "cancelled": shipping_cancelled,
            "tra_hang": shipping_return,
        },
        "shipping_status_this_month": shipping_this_month,
        "conversion": {
            "view_users": num_viewers,
            "buyer_users": num_buyers,
            "rate": conversion_rate,
        },
    })





# ========== API LOG CLICK ==========
# ========== API LOG CLICK ==========
@app.route("/api/log_click", methods=["POST"])
def log_click():
    data = request.get_json() or {}

    user_id = data.get("user_id") or "guest_user"
    laptop_id = data.get("laptop_id")
    device = data.get("device") or "android"

    if not laptop_id:
        return jsonify({"ok": False, "reason": "missing_laptop_id"}), 400

    row = {
        "user_id": user_id,
        "laptop_id": laptop_id,
        "duration_ms": None,   # ban đầu chưa có duration
    }

    supabase.table("laptop_click_logs").insert(row).execute()
    return jsonify({"ok": True})


@app.route("/api/log_view_duration", methods=["POST"])
def log_view_duration():
    data = request.get_json() or {}

    user_id = data.get("user_id") or "guest_user"
    laptop_id = data.get("laptop_id")
    duration_ms = data.get("duration_ms")

    if not laptop_id or duration_ms is None:
        return jsonify({"ok": False, "reason": "missing_fields"}), 400

    # 👇 CHỈ UPDATE, KHÔNG INSERT DÒNG MỚI
    supabase.table("laptop_click_logs") \
        .update({"duration_ms": int(duration_ms)}) \
        .eq("user_id", user_id) \
        .eq("laptop_id", laptop_id) \
        .is_("duration_ms", None) \
        .execute()

    return jsonify({"ok": True})



@app.route("/admin/api/user_stats")
def user_stats():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "missing user_id"}), 400

    # ===== 1. Log gợi ý gần đây (theo user) =====
    rec_res = (
        supabase.table("search_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    recent_recommends = rec_res.data or []

    # ===== 2. Log click / view của user =====
    click_res = (
        supabase.table("laptop_click_logs")
        .select("*")
        .eq("user_id", user_id)          # 👈 quan trọng: chỉ lấy log của user này
        .order("created_at", desc=True)
        .range(0, 9999999) 
        .execute()
    )
    click_rows = click_res.data or []

    # ===== 2a. Top 10 laptop được click nhiều nhất =====
    click_counter = Counter()
    for row in click_rows:
        lid = row.get("laptop_id")
        if not lid:
            continue
        click_counter[lid] += 1

    top_clicked = []
    for lid, total in click_counter.most_common(10):
        lap_res = (
            supabase.table("laptops")
            .select("id,name,brand,price,promo_price,image_url,description,cpu,gpu,ram_gb,storage_gb,storage_type,screen_size,weight_kg,purpose,in_stock")
            .eq("id", lid)
            .limit(1)
            .execute()
        )
        lap = (lap_res.data or [{}])[0]

        inv_res = (
            supabase.table("laptops_v")
            .select("stock_qty,sold_count,in_stock")
            .eq("id", lid)
            .limit(1)
            .execute()
        )
        inv = (inv_res.data or [{}])[0]

        merged = {**lap, **inv}  # inv override stock_qty/sold_count/in_stock nếu có

        top_clicked.append({
            "laptop_id": lid,
            **merged,
            "total": total,
        })


    # ===== 2b. Laptop user dừng lại lâu nhất =====
    stay_res = (
        supabase.table("laptop_click_logs")
        .select("laptop_id,duration_ms,created_at")   # đủ field Users.jsx dùng
        .eq("user_id", user_id)
        .gt("duration_ms", 0)                         # 👈 tránh vụ 'null'
        .order("duration_ms", desc=True)
        .limit(10)
        .execute()
    )
    stay_rows = stay_res.data or []

    longest_stay = []
    for row in stay_rows:
        lid = row["laptop_id"]
        lap_res = (
            supabase.table("laptops")
            .select("name,image_url")
            .eq("id", lid)
            .limit(1)
            .execute()
        )
        lap_list = lap_res.data or []
        lap = lap_list[0] if lap_list else None

        longest_stay.append({
            "laptop_id": lid,
            "name": lap.get("name") if lap else lid,
            "image_url": lap.get("image_url") if lap else None,
            "duration_ms": row.get("duration_ms") or 0,
            "created_at": row.get("created_at"),
        })

    # ===== 3. Giỏ hàng (tạm để trống nếu chưa join bảng orders/cart) =====
    cart_res = (
        supabase.table("carts")
        .select("id,laptop_id,quantity,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    cart_rows = cart_res.data or []

    carts = []
    for row in cart_rows:
        lid = row.get("laptop_id")
        if not lid:
            continue

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
            carts.append({
                "id": row.get("id"),
                "laptop_id": lid,
                "laptop_name": lap.get("name"),
                "image_url": lap.get("image_url"),
                "price": lap.get("price"),
                "quantity": row.get("quantity") or 1,
                "created_at": row.get("created_at"),
            })
        else:
            carts.append({
                "id": row.get("id"),
                "laptop_id": lid,
                "laptop_name": lid,
                "image_url": None,
                "price": None,
                "quantity": row.get("quantity") or 1,
                "created_at": row.get("created_at"),
            })

    # ===== 4. Phân khúc giá user hay xem =====
    price_buckets = {
        "<10tr": 0,
        "10-20tr": 0,
        "20-30tr": 0,
        "30-40tr": 0,
        ">40tr": 0,      # 👈 không có dấu cách
    }

    for r in click_rows:
        lid = r.get("laptop_id")
        if not lid:
            continue

        lap_res = (
            supabase.table("laptops")
            .select("price")
            .eq("id", lid)
            .limit(1)
            .execute()
        )
        lap_list = lap_res.data or []
        if not lap_list:
            continue

        price = lap_list[0].get("price") or 0
        if price < 10_000_000:
            price_buckets["<10tr"] += 1
        elif price < 20_000_000:
            price_buckets["10-20tr"] += 1
        elif price < 30_000_000:
            price_buckets["20-30tr"] += 1
        elif price < 30_000_000:
            price_buckets["30-40tr"] += 1
        else:
            price_buckets[">40tr"] += 1

    return jsonify({
        "recent_recommends": recent_recommends,
        "top_clicked": top_clicked,
        "longest_stay": longest_stay,
        "carts": carts,
        "price_buckets": price_buckets,
    })

# ========== API CART (MOBILE) ==========
@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    """
    Android gửi JSON:
    {
      "user_id": "...",
      "laptop_id": "...",
      "quantity": 1
    }
    → lưu vào bảng carts(id, user_id, laptop_id, quantity)
    Nếu đã có (user_id, laptop_id) thì + quantity, không tạo dòng mới.
    """
    data = request.get_json(force=True) or {}

    user_id = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    laptop_id = data.get("laptop_id")
    try:
        quantity = int(data.get("quantity", 1) or 1)
    except Exception:
        quantity = 1

    if not user_id or not laptop_id:
        return jsonify({"ok": False, "reason": "missing_user_or_laptop"}), 400

    # 1. Kiểm tra xem đã có record cart cho user+laptop này chưa
    res = (
        supabase.table("carts")
        .select("id,quantity")
        .eq("user_id", user_id)
        .eq("laptop_id", laptop_id)
        .limit(1)
        .execute()
    )

    rows = res.data or []

    if rows:
        # Đã có rồi → cập nhật quantity = quantity cũ + thêm
        row = rows[0]
        current_qty = row.get("quantity") or 0
        new_qty = current_qty + quantity
        supabase.table("carts") \
            .update({"quantity": new_qty}) \
            .match({"id": row["id"]}) \
            .execute()
    else:
        # Chưa có → insert dòng mới
        supabase.table("carts").insert({
            "user_id": user_id,
            "laptop_id": laptop_id,
            "quantity": quantity,
        }).execute()

    return jsonify({"ok": True})


# ========== API CHECKOUT ==========
@app.route("/api/cart/checkout", methods=["POST"])
def cart_checkout():
    data = request.get_json(force=True) or {}

    user_id = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    items = data.get("items") or []
    address = data.get("address") or ""
    phone_number = data.get("phone_number") or ""
    content = data.get("content") or ""
    payment_method = data.get("payment_method") or "COD"   # 👈 lấy từ app

    if not user_id:
        return jsonify({"ok": False, "reason": "missing_user_id"}), 400

    if not items:
        return jsonify({"ok": False, "reason": "no_items"}), 400

    order_items_rows = []
    total_amount = 0.0

    for it in items:
        laptop_id = it.get("laptop_id")
        if not laptop_id:
            continue

        try:
            quantity = int(it.get("quantity", 0) or 0)
        except Exception:
            quantity = 0

        try:
            price = float(it.get("price", 0) or 0)
        except Exception:
            price = 0.0

        if quantity <= 0:
            continue

        total_amount += price * quantity
        order_items_rows.append({
            "laptop_id": laptop_id,
            "quantity": quantity,
            "price": price
        })

    if not order_items_rows:
        return jsonify({"ok": False, "reason": "invalid_items"}), 400

    # 1. Tạo order (dùng supabase_admin)
    try:
        order_res = (
            supabase_admin.table("orders")
            .insert({
                "user_id": user_id,
                "total_amount": total_amount,
                "address": address,
                "phone_number": phone_number,
                "content": content,
                "status": "Chờ xác nhận",
                "payment_method": payment_method,   # 👈 nhớ có cột này trong bảng orders
            })
            .execute()
        )
    except Exception as e:
        print("checkout insert order error:", e)
        return jsonify({"ok": False, "reason": "insert_order_failed"}), 500

    order_rows = order_res.data or []
    if not order_rows:
        return jsonify({"ok": False, "reason": "order_insert_no_data"}), 500

    order_id = order_rows[0]["id"]

    # 2. Tạo order_items (dùng supabase_admin + đúng tên bảng)
        # 2. Tạo order_items (dùng supabase_admin + đúng tên bảng)
    for row in order_items_rows:
        row["order_id"] = order_id

    try:
        supabase_admin.table("order_items").insert(order_items_rows).execute()
    except Exception as e:
        print("checkout insert order_items error:", e)
        return jsonify({"ok": False, "reason": "insert_order_items_failed"}), 500

   



    # 3. Xoá khỏi carts
    try:
        laptop_ids = [row["laptop_id"] for row in order_items_rows]
        (
            supabase_admin.table("carts")
            .delete()
            .eq("user_id", user_id)
            .in_("laptop_id", laptop_ids)
            .execute()
        )
    except Exception as e:
        print("checkout delete carts error:", e)

    return jsonify({
        "ok": True,
        "order_id": order_id,
        "total_amount": total_amount
    })


@app.route("/api/cart/by_user", methods=["GET"])
def get_cart_by_user():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"ok": False, "reason": "missing_user_id"}), 400

    # ===== 1. Thử dùng supabase_admin như cũ =====
    try:
        cart_res = (
            supabase_admin.table("carts")
            .select("id,laptop_id,quantity")
            .eq("user_id", user_id)
            .execute()
        )
        cart_rows = cart_res.data or []
    except Exception as e:
        print("get_cart_by_user: supabase_admin error:", e)
        # ===== 2. Fallback: gọi REST API Supabase trực tiếp =====
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/carts",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "id,laptop_id,quantity",
                },
                headers={
                    "apikey": SUPABASE_SERVICE_ROLE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                },
                timeout=10,
            )
            r.raise_for_status()
            cart_rows = r.json() or []
        except Exception as e2:
            print("get_cart_by_user: REST fallback error:", e2)
            return jsonify({"ok": False, "reason": "supabase_error"}), 500

    if not cart_rows:
        return jsonify({"ok": True, "items": []})

    laptop_ids = [row["laptop_id"] for row in cart_rows]
    laptop_ids = list({lid for lid in laptop_ids})  # unique

    # lấy thông tin laptop
    laps_res = (
        supabase.table("laptops")
        .select("id,name,price,promo_price,image_url")
        .in_("id", laptop_ids)
        .execute()
    )
    laps = laps_res.data or []
    lap_map = {lap["id"]: lap for lap in laps}

    items = []
    for row in cart_rows:
        lap = lap_map.get(row["laptop_id"])
        if not lap:
            continue
        items.append({
            "laptop_id": lap["id"],
            "name": lap.get("name"),
            "price": lap.get("promo_price") or lap.get("price"),
            "image_url": lap.get("image_url"),
            "quantity": row.get("quantity") or 1,
        })

    return jsonify({"ok": True, "items": items})

@app.route("/api/cart/update_quantity", methods=["POST"])
def update_cart_quantity():
    data = request.get_json(force=True) or {}

    user_id = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    laptop_id = data.get("laptop_id")
    try:
        quantity = int(data.get("quantity", 0) or 0)
    except Exception:
        quantity = 0

    if not user_id or not laptop_id:
        return jsonify({"ok": False, "reason": "missing_fields"}), 400

    # Nếu số lượng <= 0 → xoá khỏi giỏ
    if quantity <= 0:
        supabase_admin.table("carts") \
            .delete() \
            .eq("user_id", user_id) \
            .eq("laptop_id", laptop_id) \
            .execute()
        return jsonify({"ok": True, "deleted": True})

    # Ngược lại, set lại quantity
    res = (
        supabase_admin.table("carts")
        .select("id")
        .eq("user_id", user_id)
        .eq("laptop_id", laptop_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []

    if rows:
        supabase_admin.table("carts") \
            .update({"quantity": quantity}) \
            .match({"id": rows[0]["id"]}) \
            .execute()
    else:
        supabase_admin.table("carts").insert({
            "user_id": user_id,
            "laptop_id": laptop_id,
            "quantity": quantity,
        }).execute()

    return jsonify({"ok": True, "deleted": False})


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


@app.route("/api/orders/by_status")
def orders_by_status():
    user_id = request.args.get("user_id")
    status = request.args.get("status")   # "Chờ xác nhận", "Chờ lấy hàng", "Chờ giao hàng", "Hoàn thành", ...

    if not user_id or not status:
        return jsonify(ok=False, reason="missing_params"), 400

    try:
        # 1. Lấy danh sách đơn
        order_res = (
            supabase_admin.table("orders")
            .select(
                """
                id,user_id,phone_number,address,content,total_amount,status,
                created_at,payment_method,updated_at,
                cancel_request_reason,cancel_request_at,
                cancel_reject_reason,cancel_reject_at,
                return_request_reason,return_request_at,
                return_reject_reason,return_reject_at
                """
            )
            .eq("user_id", user_id)
            .eq("status", status)
            .order("created_at", desc=True)
            .execute()
        )
        order_rows = order_res.data or []
        if not order_rows:
            return jsonify(ok=True, items=[]), 200

        order_ids = [o["id"] for o in order_rows]

        # 2. Lấy các dòng order_items tương ứng
        item_res = (
            supabase_admin.table("order_items")
            .select("order_id,laptop_id,quantity,price")
            .in_("order_id", order_ids)
            .execute()
        )
        item_rows = item_res.data or []

        # 3. Lấy thông tin laptop: dùng helper load_laptops_with_stock
        laptop_ids = list({r["laptop_id"] for r in item_rows}) if item_rows else []
        lap_map = {}

        if laptop_ids:
            # lấy ALL laptop đã gắn stock_qty, sold_count, in_stock
            all_laps = load_laptops_with_stock()

            # build map theo id, nhưng chỉ giữ những cái có trong laptop_ids
            for lap in all_laps:
                lid = lap.get("id")
                if lid in laptop_ids:
                    lap_map[lid] = lap
        
        profiles_map = {}
        user_ids = list({o["user_id"] for o in order_rows if o.get("user_id")})
        if user_ids:
            prof_res = (
                supabase_admin.table("profiles")
                .select("id,full_name")
                .in_("id", user_ids)
                .execute()
            )
            for p in prof_res.data or []:
                profiles_map[p["id"]] = p.get("full_name")
        # 4. Gom sản phẩm theo order_id
        items_by_order = {oid: [] for oid in order_ids}
        for row in item_rows:
            oid = row["order_id"]
            lid = row["laptop_id"]
            lap = lap_map.get(lid, {})

            items_by_order.setdefault(oid, []).append({
                "laptop_id": lid,
                "quantity": row.get("quantity") or 1,
                "price": float(row.get("price") or 0),
                "name": lap.get("name"),
                "image_url": lap.get("image_url"),
                "laptop": lap,   # FULL LaptopItem (có stock_qty, sold_count, in_stock)
            })

        # 5. Build kết quả
        orders_out = []
        for o in order_rows:
            oid = o["id"]
            user_id = o.get("user_id")
            full_name = profiles_map.get(user_id, "Khách hàng")
            orders_out.append({
                "id": oid,
                "status": o.get("status"),
                "total_amount": float(o.get("total_amount") or 0),
                "created_at": o.get("created_at"),
                "updated_at": o.get("updated_at"),
                "full_name": full_name,
                "phone_number": o.get("phone_number"),
                "address": o.get("address"),
                "content": o.get("content"),
                "payment_method": o.get("payment_method"),
                "cancel_request_reason": o.get("cancel_request_reason"),
                "cancel_request_at": o.get("cancel_request_at"),
                "cancel_reject_reason": o.get("cancel_reject_reason"),
                "cancel_reject_at": o.get("cancel_reject_at"),
                "return_request_reason": o.get("return_request_reason"),
                "return_request_at": o.get("return_request_at"),
                "return_reject_reason": o.get("return_reject_reason"),
                "return_reject_at": o.get("return_reject_at"),

                "items": items_by_order.get(oid, []),
            })

        return jsonify(ok=True, items=orders_out), 200

    except Exception as e:
        print("orders_by_status error:", e)
        return jsonify(ok=False, reason="orders_by_status_failed", detail=str(e)), 500




@app.route("/admin/api/orders")
def admin_list_orders():
    """
    Admin: list đơn với filter:
      /admin/api/orders?status=Chờ xác nhận&q=0355&page=1&page_size=20
    """
    status = request.args.get("status") or "all"   # <- nếu rỗng thì coi như all
    q = (request.args.get("q") or "").strip()
    month_param = (request.args.get("month") or "").strip()
    month_start = None
    month_end = None

    if month_param:
        try:
            # month_param dạng "2025-07"
            dt = datetime.strptime(month_param, "%Y-%m")
            y, m = dt.year, dt.month

            # tính ngày đầu tháng sau
            if m == 12:
                next_dt = datetime(y + 1, 1, 1)
            else:
                next_dt = datetime(y, m + 1, 1)

            month_start = dt.isoformat()      # "2025-07-01T00:00:00"
            month_end = next_dt.isoformat()   # "2025-08-01T00:00:00"
        except ValueError:
            # format không đúng thì bỏ qua filter tháng
            month_start = None
            month_end = None
    # --- phân trang (an toàn) ---
    try:
        page = int(request.args.get("page", 1))
    except Exception:
        page = 1
    try:
        page_size = int(request.args.get("page_size", 20))
    except Exception:
        page_size = 20

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100

    start = (page - 1) * page_size
    end = start + page_size - 1

    # base query: THÊM count="exact" để lấy tổng số đơn
    query = (
        supabase_admin.table("orders")
        .select(
            "id,user_id,phone_number,total_amount,status,created_at,content,"
            "payment_method,"
            "cancel_request_reason,cancel_request_at,"
            "cancel_reject_reason,cancel_reject_at,"
            "return_request_reason,return_request_at,"
            "return_reject_reason,return_reject_at",
            count="exact",            # 👈 QUAN TRỌNG
        )
    )

    if month_start and month_end:
        query = query.gte("created_at", month_start).lt("created_at", month_end)

    if status and status != "all":
        query = query.eq("status", status)

    if q:
        like = f"%{q}%"
        query = query.or_(
            f"id.ilike.{like},phone_number.ilike.{like},content.ilike.{like}"
        )

    try:
        resp = (
            query
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )

        rows = resp.data or []
        total = getattr(resp, "count", None)
        if total is None:
            total = len(rows)  # fallback, nhưng bình thường resp.count có giá trị

        print(
            "ADMIN_LIST_ORDERS:",
            "status=", status,
            "q=", q,
            "rows=", len(rows),
            "total=", total,
            "page=", page,
        )

        # 👇 trả thêm total, page, page_size cho frontend
        return jsonify(
            ok=True,
            items=rows,
            total=total,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        print("admin_list_orders error:", repr(e))
        return (
            jsonify(
                ok=False,
                reason="admin_list_orders_failed",
                detail=str(e),
            ),
            500,
        )



@app.route("/admin/api/orders/<order_id>")
def admin_order_detail(order_id):
    """
    Trả về:
    {
      "ok": true,
      "order": {id, status, ...},
      "items": [
         { laptop_id, name, image_url, quantity, price }
      ]
    }
    """
    try:
        # 1. order
        order_res = (
            supabase_admin.table("orders")
            .select(
                "id,user_id,phone_number,address,content,total_amount,status,created_at,payment_method,"
                "cancel_request_reason,cancel_request_at,"
                "cancel_reject_reason,cancel_reject_at,"
                "return_request_reason,return_request_at,"
                "return_reject_reason,return_reject_at"
            )
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        order_rows = order_res.data or []
        if not order_rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = order_rows[0]
         # >>> NEW: lấy full_name từ bảng profiles theo user_id
        user_id = order.get("user_id")
        customer_name = "Khách hàng"
        if user_id:
            prof_res = (
                supabase_admin.table("profiles")
                .select("full_name")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            prof_rows = prof_res.data or []
            if prof_rows:
                customer_name = prof_rows[0].get("full_name") or "Khách hàng"

        # gán vào order để FE dùng order.full_name
        order["full_name"] = customer_name
        # <<< NEW
        # 2. order_items
        item_res = (
            supabase_admin.table("order_items")
            .select("order_id,laptop_id,quantity,price")
            .eq("order_id", order_id)
            .execute()
        )
        item_rows = item_res.data or []

        laptop_ids = list({r["laptop_id"] for r in item_rows}) if item_rows else []
        lap_map = {}
        if laptop_ids:
            lap_res = (
                supabase.table("laptops")
                .select("id,name,image_url")
                .in_("id", laptop_ids)
                .execute()
            )
            for lap in lap_res.data or []:
                lap_map[lap["id"]] = lap

        items_out = []
        for r in item_rows:
            lap = lap_map.get(r["laptop_id"], {})
            items_out.append({
                "laptop_id": r["laptop_id"],
                "quantity": r.get("quantity") or 1,
                "price": float(r.get("price") or 0.0),
                "name": lap.get("name"),
                "image_url": lap.get("image_url"),
            })

        return jsonify(ok=True, order=order, items=items_out), 200

    except Exception as e:
        print("admin_order_detail error:", e)
        return jsonify(ok=False, reason="admin_order_detail_failed", detail=str(e)), 500

LOCK_STATUSES = ["Hoàn thành", "Đã hủy"]
LOCK_AFTER_DAYS = 3

# Những trạng thái được xem là đã trừ kho
DEDUCT_STOCK_STATUSES = ["Chờ giao hàng", "Hoàn thành"]  # sau này muốn thêm "Đã giao"... thì cứ nhét vô đây


def has_deducted_stock(status: str) -> bool:
    status = (status or "").strip().lower()
    return status in DEDUCT_STOCK_STATUSES


def apply_stock_for_order(order_id: str, direction: str, note_prefix: str = ""):
    # direction: 'out' = trừ kho, 'in' = cộng kho
    sign = -1 if direction == "out" else 1

    res_items = (
        supabase_admin.table("order_items")
        .select("laptop_id, quantity")
        .eq("order_id", order_id)
        .execute()
    )
    items = res_items.data or []

    if not items:
        print("apply_stock_for_order: no items for order", order_id)
        return

    for it in items:
        laptop_id = it.get("laptop_id")
        qty = it.get("quantity") or 0
        if not laptop_id or qty <= 0:
            continue

        change_qty = sign * qty

        supabase_admin.table("inventory_logs").insert(
            {
                "laptop_id": laptop_id,
                "change_qty": change_qty,
                "reason": f"{note_prefix} order {order_id}",
            }
        ).execute()



@app.route("/admin/api/orders/<order_id>/status", methods=["POST"])
def admin_update_order_status(order_id):
    data = request.get_json(force=True) or {}
    new_status = data.get("status")

    if not new_status:
        return jsonify(ok=False, reason="missing_status"), 400
    if new_status not in ORDER_STATUSES:
        return jsonify(ok=False, reason="invalid_status"), 400

    try:
        # lấy order hiện tại
        res = (
            supabase_admin.table("orders")
            .select("id,status,created_at")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]
        old_status = order.get("status") or ""
        created_at_str = order.get("created_at")

        # (tuỳ chọn) chặn đổi nếu đã khóa 3 ngày
        if old_status in LOCK_STATUSES and created_at_str:
            from datetime import datetime, timezone, timedelta

            try:
                created_at = datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                )
                now_utc = datetime.now(timezone.utc)
                if now_utc - created_at.replace(tzinfo=timezone.utc) > timedelta(
                    days=LOCK_AFTER_DAYS
                ):
                    return (
                        jsonify(
                            ok=False,
                            reason="locked_after_3_days",
                            message="Đơn đã ở trạng thái cuối hơn 3 ngày, không thể chỉnh sửa.",
                        ),
                        400,
                    )
            except Exception:
                pass

        # 👉 TÍNH TOÁN ẢNH HƯỞNG TỒN KHO TRƯỚC KHI UPDATE
        old_deduct = has_deducted_stock(old_status)
        new_deduct = has_deducted_stock(new_status)

        # cập nhật status
        supabase_admin.table("orders").update({"status": new_status}).eq(
            "id", order_id
        ).execute()

        # ====== QUY TẮC TỒN KHO ======
        # 1. Từ trạng thái chưa trừ kho -> trạng thái trừ kho  => TRỪ KHO
        if not old_deduct and new_deduct:
            try:
                apply_stock_for_order(order_id, "out", note_prefix="ship:")
            except Exception as e:
                print("apply_stock_for_order out (ship) error:", e)

        # 2. Từ trạng thái đã trừ kho -> trạng thái không trừ kho (hủy / trả) => CỘNG LẠI KHO
        elif old_deduct and not new_deduct:
            try:
                apply_stock_for_order(order_id, "in", note_prefix="rollback:")
            except Exception as e:
                print("apply_stock_for_order in (rollback) error:", e)


        return jsonify(ok=True)

    except Exception as e:
        print("admin_update_order_status error:", e)
        return (
            jsonify(
                ok=False,
                reason="update_status_failed",
                detail=str(e),
            ),
            500,
        )




@app.route("/admin/api/orders/<order_id>/reject_cancel", methods=["POST"])
def admin_reject_cancel(order_id):
    """
    Admin từ chối yêu cầu hủy:
    - Nhập lý do từ chối (reason)
    - Lưu vào cancel_reject_reason + cancel_reject_at
    - Đồng thời xóa cancel_request_reason + cancel_request_at (coi như yêu cầu hủy đã xử lý xong)
    Body JSON: { "reason": "Hàng đã đóng gói, không hủy được" }
    """
    data = request.get_json(force=True) or {}
    reason = (data.get("reason") or "").strip()

    if not reason:
        return jsonify(ok=False, reason="missing_reason"), 400

    try:
        # kiểm tra xem đơn có tồn tại + có yêu cầu hủy không
        res = (
            supabase_admin.table("orders")
            .select("id,status,cancel_request_reason")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]

        # chưa ai gửi yêu cầu hủy thì khỏi từ chối
        if not order.get("cancel_request_reason"):
            return jsonify(ok=False, reason="no_cancel_request"), 400

        # đơn đã hủy/hoàn thành thì không từ chối nữa
        if order.get("status") in ("Đã hủy", "Hoàn thành"):
            return jsonify(ok=False, reason="cannot_reject_in_this_status"), 400

        # update: lưu lý do từ chối + time, đồng thời clear request cũ
        supabase_admin.table("orders").update({
            "cancel_reject_reason": reason,
            "cancel_reject_at": datetime.now(timezone.utc).isoformat(),
            
        }).eq("id", order_id).execute()

        return jsonify(ok=True)
    except Exception as e:
        print("admin_reject_cancel error:", e)
        return jsonify(ok=False, reason="server_error", detail=str(e)), 500


@app.route("/api/orders/status_counts")
def order_status_counts():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify(ok=False, reason="missing_user_id"), 400

    try:
        # Gọi REST Supabase lấy id + status của tất cả đơn của user
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/orders",
            params={
                "user_id": f"eq.{user_id}",
                "select": "id,status",
            },
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json() or []
    except Exception as e:
        print("STATUS_COUNTS_ERROR:", repr(e))
        return jsonify(ok=False, reason="status_counts_failed", detail=str(e)), 500

    # Đếm theo chuỗi tiếng Việt đang lưu trong DB
    counts = {
        "Chờ xác nhận": 0,
        "Chờ lấy hàng": 0,
        "Chờ giao hàng": 0,
    }
    for row in rows:
        st = row.get("status")
        if st in counts:
            counts[st] += 1

    return jsonify(ok=True, counts=counts), 200

@app.route("/api/orders/request_cancel", methods=["POST"])
def request_cancel_order():
    data = request.get_json(force=True) or {}
    order_id = data.get("order_id")
    user_id = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    reason = (data.get("reason") or "").strip()

    if not order_id or not user_id or not reason:
        return jsonify(ok=False, reason="missing_fields"), 400

    # lấy order, kiểm tra chủ sở hữu và trạng thái
    try:
        res = (
            supabase_admin.table("orders")
            .select("id,user_id,status")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]
        if order["user_id"] != user_id:
            return jsonify(ok=False, reason="not_owner"), 403

        if order["status"] not in ["Chờ xác nhận", "Chờ lấy hàng"]:
            return jsonify(ok=False, reason="cannot_cancel_in_this_status"), 400

        supabase_admin.table("orders").update({
            "cancel_request_reason": reason,
            "cancel_request_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", order_id).execute()

        return jsonify(ok=True)
    except Exception as e:
        print("request_cancel_order error:", e)
        return jsonify(ok=False, reason="server_error"), 500
    
@app.route("/api/orders/confirm_received", methods=["POST"])
def confirm_received_order():
    data = request.get_json(force=True) or {}
    order_id = data.get("order_id")
    user_id = data.get("user_id")

    if not order_id or not user_id:
        return jsonify(ok=False, reason="missing_fields"), 400

    try:
        # 1. Lấy order, kiểm tra chủ sở hữu
        res = (
            supabase_admin.table("orders")
            .select("id,user_id,status")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]
        if order["user_id"] != user_id:
            return jsonify(ok=False, reason="not_owner"), 403

        # 2. Chỉ cho xác nhận khi đang "Chờ giao hàng"
        if order["status"] != "Chờ giao hàng":
            return jsonify(ok=False, reason="cannot_confirm_in_this_status"), 400

        # 3. Cập nhật sang "Hoàn thành"
        supabase_admin.table("orders").update({
            "status": "Hoàn thành",
            
        }).eq("id", order_id).execute()

        return jsonify(ok=True)
    except Exception as e:
        print("confirm_received_order error:", e)
        return jsonify(ok=False, reason="server_error"), 500

@app.route("/api/orders/request_return", methods=["POST"])
def request_return_order():
    """
    User gửi yêu cầu trả hàng.
    Body JSON: { "order_id": "...", "user_id": "...", "reason": "..." }
    Điều kiện:
      - order thuộc về user_id
      - status hiện tại = "Hoàn thành"
      - chưa có return_request_reason trước đó
    """
    data = request.get_json(force=True) or {}
    order_id = data.get("order_id")
    user_id = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    reason = (data.get("reason") or "").strip()

    if not order_id or not user_id or not reason:
        return jsonify(ok=False, reason="missing_fields"), 400

    try:
        # in log cho dễ debug
        print("REQUEST_RETURN order_id=", order_id, "user_id=", user_id, "reason=", reason)

        # lấy order để kiểm tra
        res = (
            supabase_admin.table("orders")
            .select("id,user_id,status,return_request_reason")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]

        # kiểm tra chủ sở hữu
        if order["user_id"] != user_id:
            return jsonify(ok=False, reason="not_owner"), 403

        # chỉ cho trả hàng khi đơn đã hoàn thành
        if order.get("status") != "Hoàn thành":
            return jsonify(ok=False, reason="cannot_return_in_this_status"), 400

        # nếu đã có yêu cầu trả trước đó
        if order.get("return_request_reason"):
            return jsonify(ok=False, reason="already_requested"), 400

        # update thông tin yêu cầu trả
        supabase_admin.table("orders").update({
            "return_request_reason": reason,
            "return_request_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", order_id).execute()

        return jsonify(ok=True)
    except Exception as e:
        print("request_return_order error:", e)
        return jsonify(ok=False, reason="server_error", detail=str(e)), 500



@app.route("/admin/api/orders/<order_id>/accept_return", methods=["POST"])
def admin_accept_return(order_id):
    try:
        res = (
            supabase_admin.table("orders")
            .select("id,return_request_reason,status")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]
        if not order.get("return_request_reason"):
            return jsonify(ok=False, reason="no_return_request"), 400

        # chuyển trạng thái sang "Trả hàng"
        supabase_admin.table("orders") \
            .update({"status": "Trả hàng"}) \
            .eq("id", order_id) \
            .execute()

  
        try:
            apply_stock_for_order(order_id, "in", note_prefix="return:")
        except Exception as e:
            print("apply_stock_for_order in (return) error:", e)

        return jsonify(ok=True)

    except Exception as e:
        print("admin_accept_return error:", e)
        return jsonify(ok=False, reason="server_error", detail=str(e)), 500





@app.route("/admin/api/orders/<order_id>/reject_return", methods=["POST"])
def admin_reject_return(order_id):
    data = request.get_json(force=True) or {}
    reason = (data.get("reason") or "").strip()

    if not reason:
        return jsonify(ok=False, reason="missing_reason"), 400

    try:
        res = (
            supabase_admin.table("orders")
            .select("id,status,return_request_reason")
            .eq("id", order_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify(ok=False, reason="order_not_found"), 404

        order = rows[0]

        if not order.get("return_request_reason"):
            return jsonify(ok=False, reason="no_return_request"), 400

        if order.get("status") == "Đã hủy":
            return jsonify(ok=False, reason="cannot_reject_in_this_status"), 400

        supabase_admin.table("orders").update({
            "return_reject_reason": reason,
            "return_reject_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", order_id).execute()

        return jsonify(ok=True)
    except Exception as e:
        print("admin_reject_return error:", e)
        return jsonify(ok=False, reason="server_error", detail=str(e)), 500





@app.route("/api/zalopay/create_order", methods=["POST"])
def zalopay_create_order():
    data = request.get_json(force=True) or {}

    user_id = data.get("user_id") or "guest"
    description = data.get("description") or "Thanh toán đơn hàng"

    try:
        amount = int(data.get("amount", 0) or 0)
    except Exception:
        amount = 0

    if amount <= 0:
        return jsonify(ok=False, reason="invalid_amount"), 400

    # ---- sinh apptransid unique mỗi lần ----
    app_time = int(time.time() * 1000)  # miliseconds
    app_trans_id = datetime.now().strftime("%y%m%d") + "_%06d" % random.randint(1, 999999)

    app_user = user_id[:50]  # cho chắc không quá dài

    embed_data = json.dumps({
        # tuỳ, có thể để redirecturl hoặc gì đó
        "merchantinfo": "LapTopBanChon demo"
    })
    items = json.dumps([])

    # ---- build raw string để ký MAC (đúng thứ tự trong docs) ----
    raw_data = "%s|%s|%s|%s|%s|%s|%s" % (
        ZP_APP_ID,
        app_trans_id,
        app_user,
        amount,
        app_time,
        embed_data,
        items,
    )

    mac = hmac.new(ZP_KEY1.encode(), raw_data.encode(), hashlib.sha256).hexdigest()

    order = {
        "appid": ZP_APP_ID,
        "apptransid": app_trans_id,
        "appuser": app_user,
        "apptime": app_time,
        "item": items,
        "embeddata": embed_data,
        "amount": amount,
        "description": description,
        "mac": mac,
    }

    try:
        resp = requests.post(
            ZP_ENDPOINT,
            data=order,  # 👈 FORM URLENCODED, KHÔNG PHẢI json=...
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        zp_data = resp.json()
    except Exception as e:
        print("ZP_HTTP_ERROR:", e)
        return jsonify(ok=False, reason="zalopay_http_error", detail=str(e)), 500

    print("ZP_HTTP status:", resp.status_code)
    print("ZP_HTTP body:", zp_data)

    if zp_data.get("returncode") != 1:
        # log hết ra để dễ debug
        return jsonify(
            ok=False,
            reason="zalopay_error",
            returncode=zp_data.get("returncode"),
            returnmessage=zp_data.get("returnmessage", ""),
        ), 400

    return jsonify(
        ok=True,
        zp_trans_token=zp_data.get("zptranstoken"),
        app_trans_id=app_trans_id,
    )

@app.route("/api/chatbot", methods=["POST"])
def chatbot_proxy_and_log():
    data = request.get_json(silent=True) or {}
    t0 = time.time()

    # forward sang Cloudflare Worker
    r = requests.post(
        "https://laptop-chatbot.huydao2k3.workers.dev/api/chat",
        json=data,
        timeout=12
    )

    latency_ms = int((time.time() - t0) * 1000)

    if not r.ok:
        return (r.text, r.status_code, {"Content-Type": "application/json"})

    obj = r.json()
    recs = obj.get("recommendations") or []
    result_ids = []
    for p in recs:
        pid = p.get("id") or p.get("laptop_id")
        try:
            result_ids.append(str(UUID(str(pid))))
        except:
            pass

    user_id = data.get("user_id")
    parsed_struct = obj.get("extracted") or obj.get("context") or {}

    # chỉ log khi có recs (tuỳ bạn)
    if recs and user_id:
        log_search(
            raw_query=data.get("message", ""),
            parsed_struct=parsed_struct,
            result_ids=result_ids,
            device="chatbot",
            user_id=user_id,
            query_type="chatbot",
            topk=len(result_ids),
            latency_ms=latency_ms,
            source_model="worker_chatbot_v1"
        )

    return jsonify(obj)


@app.route("/api/shop_chat/thread", methods=["GET"])
def shop_thread():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "missing user_id"}), 400

    # lấy conv mới nhất (đừng phụ thuộc status open)
    conv_res = (supabase_admin.table("shop_conversations")
        .select("id,last_admin_read_at")
        .eq("user_id", user_id)
        .order("last_message_at", desc=True)
        .limit(1)
        .execute())

    conv = conv_res.data or []
    if conv:
        conv_id = conv[0]["id"]
        last_admin_read_at = conv[0].get("last_admin_read_at")
    else:
        ins = (supabase_admin.table("shop_conversations")
            .insert({"user_id": user_id, "status": "open"})
            .execute())
        conv_id = ins.data[0]["id"]
        last_admin_read_at = None

    msgs = (supabase_admin.table("shop_messages")
        .select("id,sender_role,sender_id,content,created_at,is_recalled")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(500)
        .execute()).data or []

    return jsonify({
        "conversation_id": conv_id,
        "last_admin_read_at": last_admin_read_at,
        "messages": msgs
    })



@app.route("/api/shop_chat/send", methods=["POST"])
def shop_send():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    text = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id")

    if not user_id or not text:
        return jsonify({"error": "missing user_id/message"}), 400

    # nếu chưa có conv_id thì auto lấy/ tạo conv open
    if not conv_id:
        conv_res = (supabase_admin.table("shop_conversations")
            .select("id")
            .eq("user_id", user_id).eq("status", "open")
            .order("updated_at", desc=True).limit(1)
            .limit(1).execute())
        conv = conv_res.data or []
        conv_id = conv[0]["id"] if conv else (supabase_admin.table("shop_conversations")
            .insert({"user_id": user_id, "status": "open"})
            .execute().data[0]["id"])

    supabase_admin.table("shop_messages").insert({
        "conversation_id": conv_id,
        "sender_role": "user",
        "sender_id": user_id,
        "content": text
    }).execute()

    now_iso = datetime.now(timezone.utc).isoformat()
    supabase_admin.table("shop_conversations").update({
        "last_message_at": now_iso,
        "updated_at": now_iso
    }).eq("id", conv_id).execute()

    return jsonify({"ok": True, "conversation_id": conv_id})

@app.route("/admin/api/shop_chat/conversations", methods=["GET"])
def admin_list_convs():
    rows = (supabase_admin
            .from_("shop_conversations_admin_v")
            .select("*")
            .order("last_message_at", desc=True)
            .execute())
    return jsonify(rows.data or [])

@app.post("/admin/api/shop_chat/mark_read")
def admin_mark_read():
    data = request.get_json(silent=True) or {}
    conv_id = data.get("conversation_id")
    admin_id = data.get("admin_id")
    if not conv_id or not admin_id:
        return jsonify({"ok": False, "error": "missing params"}), 400

    p = supabase_admin.table("profiles").select("role").eq("id", admin_id).single().execute()
    if not p.data or p.data.get("role") != "admin":
        return jsonify({"ok": False, "error": "not admin"}), 403

    now_iso = datetime.now(timezone.utc).isoformat()

    supabase_admin.table("shop_conversations").update({
        "last_admin_read_at": now_iso,
        "updated_at": now_iso
    }).eq("id", conv_id).execute()

    return jsonify({"ok": True})


@app.post("/api/shop_chat/mark_read")
def user_mark_read():
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    user_id = data.get("user_id")
    if not conversation_id or not user_id:
        return jsonify({"ok": False, "error": "missing params"}), 400

    # (khuyến nghị) check conversation thuộc user_id
    conv = supabase_admin.table("shop_conversations") \
        .select("id,user_id") \
        .eq("id", conversation_id).single().execute()

    if not conv.data or conv.data.get("user_id") != user_id:
        return jsonify({"ok": False, "error": "not allowed"}), 403

    now_iso = datetime.now(timezone.utc).isoformat()

    supabase_admin.table("shop_conversations").update({
        "last_user_read_at": now_iso,
        "updated_at": now_iso
    }).eq("id", conversation_id).execute()

    return jsonify({"ok": True})

@app.get("/api/shop_chat/unread_count")
def shop_unread_count():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "missing user_id"}), 400

    conv_res = (supabase_admin.table("shop_conversations")
        .select("id,last_user_read_at")
        .eq("user_id", user_id)
        .eq("status", "open")
        .limit(1)
        .execute())
    conv = conv_res.data or []
    if not conv:
        return jsonify({"ok": True, "conversation_id": None, "unread": 0})

    conv_id = conv[0]["id"]
    last_user_read_at = conv[0].get("last_user_read_at")  # có thể None

    q = (supabase_admin.table("shop_messages")
        .select("id")
        .eq("conversation_id", conv_id)
        .eq("sender_role", "admin")
        .eq("is_recalled", False))

    if last_user_read_at:
        q = q.gt("created_at", last_user_read_at)

    rows = q.limit(500).execute().data or []
    return jsonify({"ok": True, "conversation_id": conv_id, "unread": len(rows)})


@app.route("/admin/api/shop_chat/messages", methods=["GET"])
def admin_get_messages():
    conv_id = request.args.get("conversation_id")
    if not conv_id:
        return jsonify({"error":"missing conversation_id"}), 400
    msgs = (supabase_admin.table("shop_messages")
        .select("id,sender_role,sender_id,content,created_at")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(500)
        .execute()).data or []
    return jsonify(msgs)

@app.route("/admin/api/shop_chat/send", methods=["POST"])
def admin_send_message():
    data = request.get_json(silent=True) or {}
    conv_id = data.get("conversation_id")
    admin_id = data.get("admin_id")
    text = (data.get("message") or "").strip()
    if not conv_id or not admin_id or not text:
        return jsonify({"error":"missing conversation_id/admin_id/message"}), 400

    supabase_admin.table("shop_messages").insert({
        "conversation_id": conv_id,
        "sender_role": "admin",
        "sender_id": admin_id,
        "content": text
    }).execute()
    now = datetime.now(timezone.utc).isoformat()
    supabase_admin.table("shop_conversations").update({
        "last_message_at": now,
        "updated_at": now
    }).eq("id", conv_id).execute()

    return jsonify({"ok": True})



@app.route("/api/reviews", methods=["POST"])
def api_create_or_update_review():
    data = request.get_json() or {}

    user_id    = data.get("user_id")
    if not check_user_not_locked(user_id):
        return jsonify({
            "ok": False,
            "reason": "user_locked",
            "message": "Tài khoản của bạn đã bị khóa"
        }), 403
    laptop_id  = data.get("laptop_id")
    rating_raw = data.get("rating", 0)
    content    = (data.get("content") or "").strip()
    user_name  = data.get("user_name") or "Người dùng"
    avatar     = data.get("user_avatar_url")

    # media_urls: list<string> từ app Android gửi lên (có thể rỗng)
    media_urls = data.get("media_urls") or []
    if not isinstance(media_urls, list):
        media_urls = []

    # ép rating sang int 1–5
    try:
        rating = int(float(rating_raw))
    except Exception:
        rating = 0

    if not user_id or not laptop_id or rating <= 0:
        return jsonify({"ok": False, "message": "Thiếu thông tin hoặc số sao không hợp lệ"}), 400

    # LẤY MỘT ĐƠN 'Hoàn thành' LÀM order_id (bắt buộc vì cột order_id NOT NULL)
    order_id = get_completed_order_id(user_id, laptop_id)
    if not order_id:
        return jsonify({
            "ok": False,
            "error": "NOT_DELIVERED",
            "message": "Bạn chỉ có thể đánh giá sau khi đơn hàng với sản phẩm này đã Hoàn thành."
        }), 400

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # 2. tìm review cũ (1 user – 1 review / sản phẩm)
    rv = supabase.table("reviews") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("laptop_id", laptop_id) \
        .maybe_single() \
        .execute()

    existing = getattr(rv, "data", None)  # phòng trường hợp rv = None

    if existing:
        # áp dụng rule chỉnh sửa
        if not compute_can_review(user_id, laptop_id, existing):
            return jsonify({
                "ok": False,
                "error": "EDIT_LIMIT",
                "message": "Bạn đã hết lượt chỉnh sửa hoặc đã quá thời hạn 30 ngày."
            }), 400

        edit_count = existing.get("edit_count") or 0

        update_data = {
            "rating": rating,
            "content": content,
            "user_name": user_name,
            "user_avatar_url": avatar,
            "edit_count": edit_count + 1,
            "updated_at": now_iso,
            # nếu muốn update luôn media: dùng dòng dưới
            "media_urls": media_urls
        }

        supabase.table("reviews").update(update_data) \
            .eq("id", existing["id"]).execute()
        status = "updated"
    else:
        # review mới
        insert_data = {
            "user_id": user_id,
            "laptop_id": laptop_id,
            "order_id": order_id,       # 👈 BẮT BUỘC ĐỂ KHỎI NULL
            "rating": rating,
            "content": content,
            "user_name": user_name,
            "user_avatar_url": avatar,
            "is_verified": True,        # vì đã kiểm tra đơn Hoàn thành
            "edit_count": 0,
            "helpful_count": 0,
            "created_at": now_iso,
            "media_urls": media_urls    # 👈 lưu media
        }

        supabase.table("reviews").insert(insert_data).execute()
        status = "created"

    return jsonify({"ok": True, "status": status})
def get_completed_order_id(user_id: str, laptop_id: str) -> str | None:
    """
    Trả về id của 1 đơn 'Hoàn thành' gần nhất cho user + laptop.
    Dùng để gán vào reviews.order_id (NOT NULL).
    """
    if not user_id or not laptop_id:
        return None

    resp = supabase.table("orders") \
        .select("id, status, created_at, order_items!inner(laptop_id)") \
        .eq("user_id", user_id) \
        .eq("order_items.laptop_id", laptop_id) \
        .eq("status", "Hoàn thành") \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()

    rows = resp.data or []
    if not rows:
        return None

    return rows[0].get("id")





EDIT_LIMIT = 2           # cho sửa tối đa 2 lần
EDIT_WINDOW_DAYS = 30    # trong 30 ngày kể từ lần cập nhật gần nhất


def compute_can_review(user_id: str, laptop_id: str, user_review: dict | None) -> bool:
    """
    True nếu:
    - user có ít nhất 1 đơn 'Hoàn thành' với laptop_id
    - và:
        + CHƯA có review  -> được tạo mới
        + HOẶC đã có review nhưng còn trong hạn sửa (<= 30 ngày, edit_count < 2)
    """
    if not user_id or not laptop_id:
        return False

    orders_resp = supabase.table("orders") \
        .select("id, status, created_at, order_items!inner(laptop_id)") \
        .eq("user_id", user_id) \
        .eq("order_items.laptop_id", laptop_id) \
        .eq("status", "Hoàn thành") \
        .execute()

    orders = orders_resp.data or []
    if not orders:
        return False   # chưa có đơn Hoàn thành

    # nếu chưa có review → cho review
    if not user_review:
        return True

    # rule sửa
    edit_count = user_review.get("edit_count") or 0
    if edit_count >= EDIT_LIMIT:
        return False

    updated_at_str = user_review.get("updated_at") or user_review.get("created_at")
    if not updated_at_str:
        return True    # thiếu dữ liệu thời gian thì cho sửa

    try:
        updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
    except Exception:
        return True

    now = datetime.now(timezone.utc)
    delta_days = (now - updated_at).days
    return delta_days <= EDIT_WINDOW_DAYS



@app.route("/api/reviews", methods=["GET"])
def api_get_reviews():
    laptop_id = request.args.get("laptop_id")
    user_id   = request.args.get("user_id")  # có thể None

    if not laptop_id:
        return jsonify({"error": "missing laptop_id"}), 400

    # 1. Lấy danh sách review
    resp = supabase.table("reviews") \
        .select(
            "id,user_id,user_name,user_avatar_url,"
            "laptop_id,order_id,rating,content,"
            "is_verified,edit_count,helpful_count,"
            "media_urls,"
            "created_at,updated_at"
        ) \
        .eq("laptop_id", laptop_id) \
        .order("created_at", desc=True) \
        .execute()

    rows = resp.data or []

    # 2. Tính trung bình rating & tổng số review
    ratings = [r.get("rating") for r in rows if r.get("rating") is not None]
    if ratings:
        avg_rating = round(sum(ratings) / len(ratings), 1)
    else:
        avg_rating = 0.0

    total_reviews = len(rows)

    # 3. Review của chính user (nếu có)
    user_review = None
    if user_id:
        for r in rows:
            if r.get("user_id") == user_id:
                user_review = r
                break

    # 4. Quyền review / sửa
    can_review = False
    if user_id:
        can_review = compute_can_review(user_id, laptop_id, user_review)

    return jsonify({
        "reviews": rows,
        "avg_rating": avg_rating,
        "total_reviews": total_reviews,
        "can_review": can_review,
        "user_review": user_review,
    })
@app.post("/admin/api/shop_chat/recall")
def admin_recall():
    data = request.get_json(silent=True) or {}
    admin_id = data.get("admin_id")
    message_id = data.get("message_id")

    if not admin_id or not message_id:
        return jsonify({"ok": False, "error": "missing admin_id/message_id"}), 400

    # check role admin
    p = supabase_admin.table("profiles").select("role").eq("id", admin_id).single().execute()
    if not p.data or p.data.get("role") != "admin":
        return jsonify({"ok": False, "error": "not admin"}), 403

    # lấy message để check quyền
    msg = supabase_admin.table("shop_messages") \
        .select("id, conversation_id, sender_role, sender_id, is_recalled") \
        .eq("id", message_id).single().execute()

    if not msg.data:
        return jsonify({"ok": False, "error": "message not found"}), 404

    if msg.data.get("is_recalled") is True:
        return jsonify({"ok": True}), 200

    # chỉ cho thu hồi tin do chính admin đó gửi (nếu muốn admin nào cũng thu hồi được thì bỏ điều kiện này)
    if msg.data["sender_role"] != "admin" or msg.data.get("sender_id") != admin_id:
        return jsonify({"ok": False, "error": "cannot recall this message"}), 403

    now = datetime.now(timezone.utc).isoformat()

    supabase_admin.table("shop_messages").update({
        "is_recalled": True,
        "content": "Tin nhắn đã thu hồi",
        "recalled_at": now,
        "recalled_by": admin_id
    }).eq("id", message_id).execute()

    # optional: update updated_at của conversation
    supabase_admin.table("shop_conversations").update({
        "updated_at": now
    }).eq("id", msg.data["conversation_id"]).execute()

    return jsonify({"ok": True}), 200
@app.route("/api/shop_chat/recall", methods=["POST"])
def user_recall_message():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    message_id = data.get("message_id")

    if not user_id or not message_id:
        return jsonify({"ok": False, "error": "missing user_id/message_id"}), 400

    r = (supabase_admin.table("shop_messages")
        .select("id, sender_role, sender_id, conversation_id, is_recalled")
        .eq("id", message_id)
        .single()
        .execute())

    if not r.data:
        return jsonify({"ok": False, "error": "message not found"}), 404

    if r.data.get("sender_role") != "user" or r.data.get("sender_id") != user_id:
        return jsonify({"ok": False, "error": "not allowed"}), 403

    now = datetime.now(timezone.utc).isoformat()
    supabase_admin.table("shop_messages").update({
        "is_recalled": True,
        "content": "Tin nhắn đã thu hồi",
        "recalled_at": now,
        "recalled_by": user_id
    }).eq("id", message_id).execute()

    supabase_admin.table("shop_conversations").update({
        "updated_at": now
    }).eq("id", r.data["conversation_id"]).execute()

    return jsonify({"ok": True})


def check_admin():
    """
    Return: (ok: bool, value: str)
      - ok=True  => value = admin_id
      - ok=False => value = error message
    """

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False, "Missing Authorization Bearer token"

    token = auth.split(" ", 1)[1].strip()
    if not token:
        return False, "Empty token"

    # 1) Verify token with Supabase Auth (user endpoint)
    try:
        r = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_KEY,                 # anon key OK cho /user
                "Authorization": f"Bearer {token}",
            },
            timeout=10,
        )
        if not r.ok:
            return False, f"Invalid token ({r.status_code})"

        user = r.json() or {}
        admin_id = user.get("id")
        if not admin_id:
            return False, "Token valid but missing user id"
    except Exception as e:
        return False, f"Auth verify error: {e}"

    # 2) Check role in profiles
    try:
        prof = (
            supabase_admin.table("profiles")
            .select("role,is_locked")
            .eq("id", admin_id)
            .single()
            .execute()
        )
        pdata = prof.data or {}
        role = (pdata.get("role") or "").strip().lower()
        if role != "admin":
            return False, "Not admin"

        # (tuỳ chọn) chặn admin bị khoá
        if pdata.get("is_locked") is True:
            return False, "Admin account is locked"

        return True, admin_id
    except Exception as e:
        return False, f"Profile check error: {e}"
@app.route("/admin/api/users/lock", methods=["POST"])
def lock_user():
    ok, admin_id = check_admin()
    if not ok:
        return jsonify({"error": admin_id}), 403

    data = request.json
    user_id = data.get("user_id")
    reason = data.get("reason", "Vi phạm chính sách")

    if not user_id:
        return jsonify({"error": "Thiếu user_id"}), 400

    supabase.table("profiles").update({
        "is_locked": True,
        "locked_at": "now()",
        "locked_reason": reason
    }).eq("id", user_id).execute()

    return jsonify({"success": True})
@app.route("/admin/api/users/unlock", methods=["POST"])
def unlock_user():
    ok, admin_id = check_admin()
    if not ok:
        return jsonify({"error": admin_id}), 403

    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"error": "Thiếu user_id"}), 400

    supabase.table("profiles").update({
        "is_locked": False,
        "locked_at": None,
        "locked_reason": None
    }).eq("id", user_id).execute()

    return jsonify({"success": True})
def check_user_not_locked(user_id):
    profile = (
        supabase
        .table("profiles")
        .select("is_locked")
        .eq("id", user_id)
        .single()
        .execute()
        .data
    )
    return not profile.get("is_locked", False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
