from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

def _to_vnd(s: Optional[str]) -> Optional[int]:
    if not s: return None
    t = str(s).lower().strip()
    t = (t.replace("vnđ","").replace("vnd","").replace("đ","")
           .replace("triệu","0000000").replace("tr","0000000")
           .replace(".","").replace(","," "))
    digits = "".join(ch for ch in t if ch.isdigit())
    return int(digits) if digits else None

class ActionSearchLaptop(Action):
    def name(self) -> Text:
        return "action_search_laptop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        usage = (tracker.get_slot("usage") or "").strip()
        brand = (tracker.get_slot("brand") or "").strip()
        price_max_raw = (tracker.get_slot("price_max") or "").strip()
        price_max = _to_vnd(price_max_raw)

        # hỏi dần nếu thiếu
        if not usage:
            dispatcher.utter_message(response="utter_ask_usage"); return []
        if price_max is None:
            dispatcher.utter_message(response="utter_ask_price_max"); return []
        if not brand:
            dispatcher.utter_message(response="utter_ask_brand"); return []

        # brand_any → bỏ filter_brand
        brand_any_values = {"hãng nào cũng được","tuỳ bạn chọn","không quan trọng hãng","sao cũng được","tùy bạn chọn","không cần hãng"}
        filter_brand = None if brand.lower() in brand_any_values else brand

        payload: Dict[str, Any] = {
            "query": tracker.latest_message.get("text", ""),
            "device": "rasa",
            "max_price": price_max,
        }
        if filter_brand:
            payload["filter_brand"] = filter_brand

        usage_map = {
            "văn phòng": "office",
            "học online": "office",
            "chơi game": "gaming",
            "đồ hoạ": "design",
            "đồ họa": "design",
            "thiết kế": "design",
        }
        payload["usage"] = usage_map.get(usage.lower(), usage.lower())

        try:
            r = requests.post("http://192.168.100.237:5000/api/recommend", json=payload, timeout=8)
            if r.status_code != 200:
                dispatcher.utter_message(text="Vậy bạn chờ mình chút nhé!"); return []
            items = (r.json() or {}).get("items", [])[:10]
            if not items:
                dispatcher.utter_message(text="Không tìm thấy laptop phù hợp với tiêu chí hiện tại.")
                dispatcher.utter_message(response="utter_followup_more"); return []
            lines = ["Mình gợi ý bạn mấy mẫu này:"]
            for it in items:
                name = it.get("name","Laptop"); brand_res = it.get("brand","")
                price = it.get("promo_price") or it.get("price") or 0
                lines.append(f"- {name} ({brand_res}) ~ {price:,.0f} đ")
            dispatcher.utter_message("\n".join(lines))
            dispatcher.utter_message(response="utter_followup_more")
        except Exception:
            dispatcher.utter_message(text="Vậy bạn chờ mình chút nhé!")
        return []
