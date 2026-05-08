"""Query preprocessing for STT -> RAG pipeline."""

from __future__ import annotations

import re

STT_TRANSLITERATION_DICT: dict[str, str] = {
    "agrifaro": "అగ్రి ఫెర్రో",
    "agri faro": "అగ్రి ఫెర్రో",
    "pheromone": "ఫెరమోన్",
    "pandu eagle": "పండు ఈగ",
    "pandu eagles": "పండు ఈగ",
    "kattera purugu": "కత్తెర పురుగు",
    "katar purugu": "కత్తెర పురుగు",
    "kanda tolichu purugu": "కాండం తొలిచే పురుగు",
    "kanda tolichu": "కాండం తొలిచే",
    "sticky trap": "స్టిక్కీ ట్రాప్",
    "sticky traps": "స్టిక్కీ ట్రాప్స్",
    "funnel trap": "ఫన్నెల్ ట్రాప్",
    "funnel traps": "ఫన్నెల్ ట్రాప్స్",
    "agrifero": "అగ్రి ఫెర్రో",
    "agri fhero": "అగ్రి ఫెర్రో",
    "company": "కంపెనీ",
    "compani": "కంపెనీ",
    "kompani": "కంపెనీ",
    "founder": "ఫౌండర్స్",
    "founders": "ఫౌండర్స్",
    "who founded": "ఎవరు పెట్టారు",
    "who started": "ఎవరు పెట్టారు",
    "who made": "ఎవరు పెట్టారు",
    "evaru pettaru": "ఎవరు పెట్టారు",
    "evarandi": "ఎవరు",
    "evaru": "ఎవరు",
    "evvaru": "ఎవరు",
    "eppudu": "ఎప్పుడు",
    "eppudandi": "ఎప్పుడు",
    "pettaru": "పెట్టారు",
    "tayar chesaru": "పెట్టారు",
    "tayarichesaru": "పెట్టారు",
    "tayarchesaru": "పెట్టారు",
    "tayaru chesaru": "పెట్టారు",
    "years": "సంవత్సరాలు",
    "year": "సంవత్సరం",
    "samvatsaralu": "సంవత్సరాలు",
    "samvatsaram": "సంవత్సరం",
    "peco": "అగ్రి ఫెర్రో",
    "peko": "అగ్రి ఫెర్రో",
    "feco": "అగ్రి ఫెర్రో",
    "phero": "ఫెర్రో",
    "lure bait": "లూర్",
    "address": "అడ్రస్",
    "contact": "కాంటాక్ట్",
    "contact details": "కాంటాక్ట్ వివరాలు",
    "phone": "ఫోన్",
    "phone number": "ఫోన్ నంబర్",
    "website": "వెబ్‌సైట్",
    "web site": "వెబ్‌సైట్",
    "pincode": "పిన్ కోడ్",
    "pin code": "పిన్ కోడ్",
    "chirunama": "చిరునామా",
    "chiru nama": "చిరునామా",
    "chiroona": "చిరునామా",
    "chiroonama": "చిరునామా",
    "फोन नंबर": "ఫోన్ నంబర్",
    "एड्रेस": "అడ్రస్",
    "कॉन्टैक्ट": "కాంటాక్ట్",
    "पता": "చిరునామా",
}


def preprocess_stt_query(query: str) -> str:
    raw = query.lower()
    q = raw
    for wrong, correct in STT_TRANSLITERATION_DICT.items():
        q = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, q)

    founder_intent = any(x in q for x in ["founder", "ఫౌండర్స్", "ఎవరు పెట్టారు", "ఎవరు"])
    year_intent = any(x in q for x in ["year", "years", "సంవత్సర", "ఎప్పుడు", "since"])
    contact_intent = any(
        x in q
        for x in [
            "అడ్రస్",
            "కాంటాక్ట్",
            "ఫోన్",
            "నంబర్",
            "చిరునామా",
            "వెబ్‌సైట్",
            "పిన్ కోడ్",
            "address",
            "contact",
            "phone",
            "website",
            "pincode",
            "సంప్రదించ",
            "फोन",
            "नंबर",
            "एड्रेस",
            "कॉन्टैक्ट",
            "पता",
        ]
    )

    if founder_intent:
        q += " ఫౌండర్స్ సంస్థాపకులు"

    if year_intent:
        q += " సంస్థ స్థాపన ప్రారంభం అనుభవం సంవత్సరాలు"

    if contact_intent:
        q += " సంస్థను ఎలా సంప్రదించాలి అడ్రస్ కాంటాక్ట్ వివరాలు ఫోన్ నంబర్ వెబ్‌సైట్ చిరునామా"

    return " ".join(q.split())
