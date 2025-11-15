# actions.py
from typing import Any, Text, Dict, List, Optional, Tuple
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import requests, logging, time, json, re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =========================
# Helpers
# =========================
def _looks_like_money(text: Optional[str]) -> bool:
    if not text:
        return False
    t = str(text).lower().strip()
    return bool(re.search(r"\d+\s*(tr|triệu)\b", t)) or bool(re.search(r"\b\d{6,}\b", t))

def _to_vnd(price_text: Optional[str]) -> Optional[int]:
    if not price_text:
        return None
    s = str(price_text).lower()
    s = s.replace("vnđ", "").replace("vnd", "").replace("đ", "")
    s = s.replace(",", " ").replace(".", " ").replace("triệu", "tr")
    nums_tr = re.findall(r"(\d+)\s*tr\b", s)
    if nums_tr:
        return max(int(n) * 1_000_000 for n in nums_tr)
    nums_raw = re.findall(r"\b(\d{6,})\b", s)
    if nums_raw:
        return max(int(n) for n in nums_raw)
    m = re.search(r"(dưới|toi|tối đa|max|không quá|<=)\s*(\d+)", s)
    if m:
        return int(m.group(2)) * 1_000_000
    return None

def _is_brand_any(text: Optional[str]) -> bool:
    if not text:
        return False
    t = str(text).strip().lower()
    return t in {
        "any","không quan trọng","khong quan trong","tùy","tuy","tuỳ",
        "tuỳ bạn chọn","hãng nào cũng được","sao cũng được","miễn ổn là được",
        "không cần hãng","không ưu tiên hãng","không quan tâm thương hiệu",
    }

def _normalize_usage(v: Optional[str]) -> str:
    if not v:
        return ""
    m = {
        "văn phòng":"office","lam viec":"office","làm việc":"office",
        "học online":"office","hoc online":"office",
        "chơi game":"gaming","choi game":"gaming",
        "đồ hoạ":"design","đồ họa":"design","do hoa":"design","thiết kế":"design",
    }
    v2 = v.lower().strip()
    return m.get(v2, v2)

def _clean_slots_from_noise(latest_text: str, raw_brand: Optional[str], raw_price: Optional[str]) -> Tuple[str, Optional[int]]:
    """
    - Ưu tiên dùng raw_price nếu hợp lệ (và khác latest_text).
    - Chỉ parse từ latest_text nếu latest_text trông giống tiền.
    - Brand mà là cả câu/tiền -> 'any'. Chuẩn hóa 'không quan trọng' -> 'any'.
    """
    brand_clean = (raw_brand or "").strip()
    price_max: Optional[int] = None

    if raw_price and raw_price != latest_text:
        price_max = _to_vnd(raw_price)

    if price_max is None and _looks_like_money(latest_text):
        price_max = _to_vnd(latest_text)

    if (not brand_clean) or brand_clean == latest_text or _looks_like_money(brand_clean):
        brand_clean = "any"
    if _is_brand_any(brand_clean):
        brand_clean = "any"

    return brand_clean, price_max

# =========================
# Actions
# =========================
class ActionClearSlots(Action):
    def name(self) -> Text:
        return "action_clear_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Xóa các slot tìm kiếm
        return [
            SlotSet("usage", None),
            SlotSet("price_max", None),
            SlotSet("brand", None),
        ]

class ActionSetBrandAny(Action):
    """Chỉ set 'any' khi thực sự là 'không quan trọng'.
       Nếu có entity brand cụ thể (MSI, Dell, ...), ưu tiên set đúng hãng đó.
    """
    def name(self) -> Text:
        return "action_set_brand_any"

    def run(self, dispatcher, tracker, domain):
        # lấy entity brand ở tin nhắn cuối
        ents = (tracker.latest_message or {}).get("entities", []) or []
        brand_ent = None
        for e in ents:
            if e.get("entity") == "brand":
                brand_ent = (e.get("value") or "").strip()
                break

        # nếu user nói cụ thể 1 hãng -> set hãng đó
        if brand_ent and brand_ent.lower() not in {"any", "không quan trọng", "khong quan trong"}:
            return [SlotSet("brand", brand_ent), FollowupAction("action_search_laptop")]

        # còn lại mới set any
        return [SlotSet("brand", "any"), FollowupAction("action_search_laptop")]


class ActionSearchLaptop(Action):
    def name(self) -> Text:
        return "action_search_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        start = time.perf_counter()

        latest = tracker.latest_message or {}
        latest_text = latest.get("text", "") or ""

        usage_raw = (tracker.get_slot("usage") or "").strip()
        brand_raw = (tracker.get_slot("brand") or "").strip() if tracker.get_slot("brand") else ""
        price_raw = (tracker.get_slot("price_max") or "").strip() if tracker.get_slot("price_max") else ""

        # KHỞI TẠO events SỚM
        events: List[Dict[str, Any]] = []

        # Làm sạch
        brand_clean, price_max = _clean_slots_from_noise(latest_text, brand_raw, price_raw)
        usage_norm = _normalize_usage(usage_raw)

        # Ghi ngược slot CHỈ khi có giá trị hợp lệ & khác cũ
        if brand_clean and brand_clean != brand_raw:
            events.append(SlotSet("brand", brand_clean))

        # Nếu price_max parse được, đồng bộ về slot dạng số
        if price_max is not None and (price_raw or "") != str(price_max):
            events.append(SlotSet("price_max", str(price_max)))
            price_raw = str(price_max)  # để dưới sử dụng nhất quán

        logger.debug("[RASA] slots_after_clean usage=%r brand=%r price_max=%r", usage_norm, brand_clean, price_max)

        # Hỏi bù theo thứ tự: usage -> price -> brand
        if not usage_norm:
            dispatcher.utter_message(response="utter_ask_usage")
            return events + [FollowupAction("action_listen")]

        # Giá: nếu slot đang rác/chuỗi không số & chưa parse được → hỏi lại
        price_is_numeric = price_raw.isdigit()
        if not price_is_numeric and price_max is None:
            dispatcher.utter_message(response="utter_ask_price_max")
            return events + [FollowupAction("action_listen")]

        # Brand: cần hỏi khi trống, hoặc bị normalize thành 'any' nhưng raw không phải 'any'
        need_ask_brand = False
        if not brand_raw:
            need_ask_brand = True
        elif brand_clean == "any" and not _is_brand_any(brand_raw):
            need_ask_brand = True

        if need_ask_brand:
            dispatcher.utter_message(response="utter_ask_brand")
            return events + [FollowupAction("action_listen")]

        # Chuẩn bị payload
        max_price_val: Optional[int] = None
        if price_is_numeric:
            try:
                max_price_val = int(price_raw)
            except Exception:
                max_price_val = price_max
        else:
            max_price_val = price_max

        payload: Dict[str, Any] = {
            "query": latest_text,
            "device": "rasa",
            "max_price": max_price_val,
            "usage": usage_norm,
        }
        if brand_clean and not _is_brand_any(brand_clean):
            payload["filter_brand"] = brand_clean

        logger.debug("[RASA→FLASK] payload_out=%s", json.dumps(payload, ensure_ascii=False))

        try:
            r = requests.post("http://192.168.100.237:5000/api/recommend", json=payload, timeout=8)
            logger.debug("[FLASK] status=%s elapsed=%.1fms", r.status_code, (time.perf_counter()-start)*1000)

            if r.status_code != 200:
                logger.error("[FLASK] Non-200: code=%s body=%r", r.status_code, r.text[:400])
                dispatcher.utter_message(text="Server gợi ý bị lỗi, bạn thử lại sau nha.")
                return events + [FollowupAction("action_listen")]

            data = r.json()
            items = (data.get("items") or [])[:10]
            logger.debug("[FLASK] items_len=%d debug=%s", len(items), data.get("debug"))

            if not items:
                dispatcher.utter_message(text="Không tìm thấy laptop phù hợp với tiêu chí hiện tại.")
                dispatcher.utter_message(response="utter_followup_more")
                return events + [FollowupAction("action_listen")]

            lines = ["Mình gợi ý bạn mấy mẫu này:"]
            for it in items:
                name = it.get("name", "Laptop")
                brand_res = it.get("brand", "")
                price = it.get("promo_price") or it.get("price") or 0
                try:
                    price_fmt = f"{int(price):,}".replace(",", ".")
                except Exception:
                    price_fmt = str(price)
                lines.append(f"- {name} ({brand_res}) ~ {price_fmt} đ")

            dispatcher.utter_message("\n".join(lines))
            dispatcher.utter_message(response="utter_after_results")
            dispatcher.utter_message(response="utter_followup_more")
            return events + [FollowupAction("action_listen")]

        except Exception:
            logger.exception("[FLASK] Unexpected error calling /api/recommend")
            dispatcher.utter_message(text="Server gợi ý bị lỗi, bạn thử lại sau nha.")
            return events + [FollowupAction("action_listen")]
