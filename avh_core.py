import os
import sys
import json
import glob
import re
import requests
import urllib.parse
import time
import math
import html
import subprocess
import collections
from datetime import datetime
import zhconv

# ==============================================================================
# AVH Genesis Engine (V57.0 本體主軸守門版 - Ontology-first, no hard-coded topic lexicon)
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "avh_manifest.json")

OLLAMA_MODEL_NAME = "gemma4"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

FULLTEXT_NUM_CTX = 32768
SUMMARY_NUM_CTX = 32768
BACKGROUND_NUM_CTX = 8192
CLASSIFY_NUM_CTX = 6144

RETRIEVAL_ROWS_PER_PROBE = 8
PROBE_WORD_MIN = 6
PROBE_WORD_MAX = 24
DOC_CAPTURE_THRESHOLD = 0.08
REQUIRE_ABSTRACT_FOR_FINAL = True
MAX_GLOBAL_FINAL = 8
MAX_SHADOW_LOG = 12

ABSTRACT_CACHE = {}

print(f"🧠 [載入本地觀測核心] 啟動 V57.0 本體主軸守門版 (引擎: {OLLAMA_MODEL_NAME})...")

if not os.path.exists(MANIFEST_PATH):
    print(f"⚠️ 遺失底層定義檔：{MANIFEST_PATH}，系統終止觀測。")
    sys.exit(1)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

DIMENSION_KEYS = list(MANIFEST["dimensions"].keys())

STOP_WORDS = {
    "the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", "this",
    "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your",
    "all", "have", "new", "we", "an", "was", "can", "will", "via", "using", "based",
    "proposing", "propose", "study", "approach", "method", "methods", "paper",
    "research", "article", "system", "current", "their", "into", "than", "such"
}

# ------------------------------------------------------------------------------
# Text/vector utilities
# ------------------------------------------------------------------------------

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_crossref_abstract(raw_abstract):
    return normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", raw_abstract or "")))


def clamp(value, low, high):
    return max(low, min(high, value))


def dim_label(key):
    return MANIFEST["dimensions"][key]["layer"]


def sign_to_binary(scores_by_key):
    return "".join("1" if scores_by_key[k] > 0 else "0" for k in DIMENSION_KEYS)


def signed_score_to_side(score):
    return "離群突破（虛部/sin）" if score > 0 else "合群守成（實部/cos）"


def enforce_score(value, field_name):
    try:
        return clamp(int(round(float(value))), -100, 100)
    except Exception:
        raise ValueError(f"{field_name} 分數異常：{value}")


def enforce_confidence(value, field_name):
    try:
        return clamp(int(round(float(value))), 0, 100)
    except Exception:
        raise ValueError(f"{field_name} 置信度異常：{value}")


def angle_from_cosine(cos_val):
    return math.degrees(math.acos(clamp(cos_val, -1.0, 1.0)))


def proximity_from_scores(user_score, background_score):
    diff = abs(user_score - background_score)
    return round(max(0.0, 100.0 - diff / 2.0), 1)


def classify_relation(user_score, background_score):
    if abs(background_score) < 10:
        return "弱耦合"
    if user_score == 0 and background_score == 0:
        return "中性"
    if user_score * background_score < 0:
        return "反向"
    mag_u, mag_b = abs(user_score), abs(background_score)
    if abs(mag_u - mag_b) <= 10:
        return "同向近似"
    return "同向"


def compact_title(title, max_len=72):
    title = normalize_whitespace(title)
    return title if len(title) <= max_len else title[: max_len - 1] + "…"


def simple_escape(text):
    if not text:
        return ""
    out = str(text)
    for src, dst in [
        ("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$") ,
        ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}")
    ]:
        out = out.replace(src, dst)
    return out


def markdown_to_latex(text):
    out = []
    for line in str(text).splitlines():
        if line.startswith("### "):
            out.append(f"\\subsubsection{{{simple_escape(line[4:])}}}")
        elif line.startswith("## "):
            out.append(f"\\subsection{{{simple_escape(line[3:])}}}")
        elif line.startswith("# "):
            out.append(f"\\section{{{simple_escape(line[2:])}}}")
        else:
            out.append(simple_escape(line))
    return "\n".join(out)


def get_text_vector(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    return dict(collections.Counter(filtered))


def compute_dict_cosine(d1, d2):
    intersection = set(d1.keys()) & set(d2.keys())
    numerator = sum(d1[x] * d2[x] for x in intersection)
    sum1 = sum(v ** 2 for v in d1.values())
    sum2 = sum(v ** 2 for v in d2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator


def build_dimensions_prompt():
    payload = [
        {
            "key": k,
            "zh_label": MANIFEST["dimensions"][k]["layer"],
            "sin_def": MANIFEST["dimensions"][k]["sin_def"],
            "cos_def": MANIFEST["dimensions"][k]["cos_def"]
        }
        for k in DIMENSION_KEYS
    ]
    return json.dumps(payload, ensure_ascii=False)


def unique_list(seq, limit=None):
    seen = set()
    out = []
    for item in seq:
        s = normalize_whitespace(item)
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def normalize_statement(stmt):
    s = normalize_whitespace(stmt)
    s = s.strip("`\"' ")
    s = re.sub(r"\s+", " ", s)
    return s


def is_generic_probe_statement(stmt):
    s = normalize_statement(stmt).lower()
    bad_patterns = [
        r"^we must\b", r"^the system requires\b", r"^a study on\b",
        r"^proposing a new\b", r"^this paper\b", r"^we propose\b",
        r"^an approach to\b", r"^it is necessary to\b"
    ]
    return any(re.search(p, s) for p in bad_patterns)


def is_valid_probe_statement(stmt):
    s = normalize_statement(stmt)
    word_count = len(re.findall(r'\b[a-zA-Z]+\b', s))
    if word_count < PROBE_WORD_MIN or word_count > PROBE_WORD_MAX:
        return False
    if is_generic_probe_statement(s):
        return False
    return True


def is_similar_title(t1, t2):
    w1 = set(re.findall(r'\w+', str(t1).lower()))
    w2 = set(re.findall(r'\w+', str(t2).lower()))
    if not w1 or not w2:
        return False
    return (len(w1 & w2) / len(w1 | w2)) > 0.5


# ------------------------------------------------------------------------------
# LLM comms
# ------------------------------------------------------------------------------

def call_local_llm(messages, json_mode=False, temperature=0.0, num_ctx=8192):
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx}
    }
    if json_mode:
        payload["format"] = "json"

    payload_size = len(json.dumps(payload, ensure_ascii=False))
    print(f"   ↳ ⚡ [物理探測] 即將注入資訊熵：{payload_size} 字元。上下文視窗：{num_ctx}...")
    start_time = time.time()

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=420)
        response.raise_for_status()
        elapsed = time.time() - start_time
        print(f"   ↳ 🟢 [觀測完成] 耗時 {elapsed:.1f} 秒，實體質量成功顯化。")
        return response.json()["message"]["content"]
    except requests.exceptions.Timeout:
        print("\n❌ [邊界破裂] 運算超過 420 秒！資訊熵過載導致引擎超時。")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ [邊界破裂] 無法連線至 Ollama。模型可能因 VRAM 溢出而崩潰。")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 本地推演發生網路或通訊錯誤: {e}")
        raise


def parse_llm_json(response_text):
    text = str(response_text).strip()
    if not text:
        raise ValueError("LLM 回傳空白。")

    fence = chr(96) * 3
    pattern = fence + r"(?:json)?\s*(.*?)\s*" + fence
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("找不到 JSON 起始符號 '{'")

    end_idx, depth, in_string, escape = -1, 0, False, False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        escape = True if ch == "\\" and not escape else False

    if end_idx == -1:
        raise ValueError("找不到完整 JSON 結尾。")

    clean_json = text[start_idx:end_idx + 1]
    try:
        return json.loads(clean_json, strict=False)
    except Exception:
        return json.loads(re.sub(r"(?<!\\)\n", " ", clean_json), strict=False)


# ------------------------------------------------------------------------------
# External abstract augmentation
# ------------------------------------------------------------------------------

def reconstruct_openalex_abstract(inv_idx):
    if not inv_idx or not isinstance(inv_idx, dict):
        return ""
    pos_map = {}
    for word, positions in inv_idx.items():
        for pos in positions:
            pos_map[pos] = word
    if not pos_map:
        return ""
    max_pos = max(pos_map.keys())
    words = [pos_map.get(i, "") for i in range(max_pos + 1)]
    return normalize_whitespace(" ".join(words))


def fetch_crossref_abstract_by_doi(doi):
    try:
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
        r = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/57.0"}, timeout=20)
        r.raise_for_status()
        abstract = clean_crossref_abstract(r.json().get("message", {}).get("abstract", ""))
        return abstract, "crossref_doi" if abstract else ("", "")
    except Exception:
        return "", ""


def fetch_openalex_abstract_by_doi(doi):
    try:
        doi_url = f"https://doi.org/{doi}"
        url = f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(doi_url, safe=':/')}&per-page=1"
        r = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/57.0"}, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return "", ""
        abstract = reconstruct_openalex_abstract(results[0].get("abstract_inverted_index", {}))
        return abstract, "openalex" if abstract else ("", "")
    except Exception:
        return "", ""


def fetch_semanticscholar_abstract_by_doi(doi):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,abstract"
        r = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/57.0"}, timeout=20)
        r.raise_for_status()
        abstract = normalize_whitespace(r.json().get("abstract", ""))
        return abstract, "semanticscholar" if abstract else ("", "")
    except Exception:
        return "", ""


def fetch_external_abstract(doi):
    if not doi or doi == "Unknown":
        return "", ""
    if doi in ABSTRACT_CACHE:
        return ABSTRACT_CACHE[doi]
    for fn in [fetch_crossref_abstract_by_doi, fetch_openalex_abstract_by_doi, fetch_semanticscholar_abstract_by_doi]:
        abstract, source = fn(doi)
        if abstract:
            ABSTRACT_CACHE[doi] = (abstract, source)
            return abstract, source
    ABSTRACT_CACHE[doi] = ("", "")
    return "", ""


# ------------------------------------------------------------------------------
# Ontology-first profile extraction
# ------------------------------------------------------------------------------

def evaluate_user_profile(raw_text):
    sys_prompt = f"""
你是一台「本體主軸守門型學術觀測儀」。
任務不是抓漂亮關鍵字，而是先從全文抽出這篇文章自己的本體結構，再依此生成外發探針。
禁止把作者拿來比喻或批判的語言，誤當成主題本體。
只能回傳合法 JSON。

維度定義：
{build_dimensions_prompt()}

規則：
1. ontology_profile.topic_world：作者真正討論的世界是什麼。
2. ontology_profile.core_object：真正被觀測/定位/評估的對象是什麼。
3. ontology_profile.mechanism_language：作者用來描述的機制語言，可包含數學、物理、拓樸、治理語彙。
4. ontology_profile.supporting_anchors：作者明確支持、建構或倡導的主題錨點。
5. ontology_profile.metaphor_anchors：作者借來描述但不可主導外部檢索的比喻/隱喻語言。
6. ontology_profile.forbidden_or_opposed：作者正在批判、排除或推翻的概念。
7. probe_bundle_8：輸出 8 個外發探針，每個都必須有 role 與 statement。role 只能是 topic / object / mechanism / bridge。至少各角色要出現。英文句，每句 <= 24 個英文單字。
8. primary_statement：最能代表全文骨架的一句英文核心論述。
9. retrieval_signature_en：拿去和外部文獻比對時最穩定的全文簽名。英文一句。
10. implementation_signals / application_signals：保留全文實作與輸出證據。
11. 六維 score/confidence/reason 依全文量化。
12. 禁止神祕學語彙。禁止把批判對象當作者支持內容。

JSON 結構：
{{
  "ontology_profile": {{
    "topic_world": "...",
    "core_object": "...",
    "mechanism_language": ["..."],
    "supporting_anchors": ["..."],
    "metaphor_anchors": ["..."],
    "forbidden_or_opposed": ["..."]
  }},
  "primary_statement": "...",
  "retrieval_signature_en": "...",
  "probe_bundle_8": [
    {{"role": "topic", "statement": "..."}},
    {{"role": "object", "statement": "..."}},
    {{"role": "mechanism", "statement": "..."}},
    {{"role": "bridge", "statement": "..."}}
  ],
  "implementation_signals": ["..."],
  "application_signals": ["..."],
  "academic_fingerprint": "...",
  "value_intent_score": 85,
  "value_intent_confidence": 92,
  "value_intent_reason": "說明",
  "governance_score": 70,
  "governance_confidence": 88,
  "governance_reason": "說明",
  "cognition_score": 60,
  "cognition_confidence": 90,
  "cognition_reason": "說明",
  "architecture_score": 40,
  "architecture_confidence": 85,
  "architecture_reason": "說明",
  "expansion_score": 30,
  "expansion_confidence": 80,
  "expansion_reason": "說明",
  "application_score": -10,
  "application_confidence": 75,
  "application_reason": "說明"
}}
""".strip()

    print("🕸️ [階段 1] 本體主軸直讀：先抽 ontology profile，再釋放分角色探針...")
    user_prompt = f"【全文開始】\n{raw_text}\n【全文結束】\n⚠️ 只能輸出 JSON。"

    res = parse_llm_json(
        call_local_llm(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            json_mode=True,
            num_ctx=FULLTEXT_NUM_CTX
        )
    )

    ontology_profile = res.get("ontology_profile", {}) if isinstance(res.get("ontology_profile", {}), dict) else {}
    raw_primary = normalize_statement(res.get("primary_statement", ""))
    raw_signature = normalize_statement(res.get("retrieval_signature_en", ""))
    raw_bundle = res.get("probe_bundle_8", [])

    probe_bundle = []
    role_counter = collections.Counter()
    if isinstance(raw_bundle, list):
        for item in raw_bundle:
            if not isinstance(item, dict):
                continue
            role = normalize_whitespace(item.get("role", "")).lower()
            statement = normalize_statement(item.get("statement", ""))
            if role not in {"topic", "object", "mechanism", "bridge"}:
                continue
            if not is_valid_probe_statement(statement):
                print(f"   ↳ [過濾剔除] Probe 格式不合或過度口號化：{statement}")
                continue
            probe_bundle.append({"role": role, "statement": statement})
            role_counter[role] += 1

    if raw_primary and is_valid_probe_statement(raw_primary):
        if raw_primary.lower() not in {p["statement"].lower() for p in probe_bundle}:
            probe_bundle.insert(0, {"role": "bridge", "statement": raw_primary})
            role_counter["bridge"] += 1

    probe_bundle = unique_probe_bundle(probe_bundle)

    if not probe_bundle:
        print("   ↳ ⚠️ [容錯介入] 全文直讀未產生有效 probe，啟動標題淬取...")
        fallback = fallback_statement_from_text(raw_text)
        probe_bundle = [{"role": "topic", "statement": fallback}]

    primary_statement = raw_primary if raw_primary and is_valid_probe_statement(raw_primary) else probe_bundle[0]["statement"]
    retrieval_signature = raw_signature if raw_signature else primary_statement

    print(f"   ↳ 🎯 [本體結構] topic_world = {normalize_whitespace(ontology_profile.get('topic_world', '未抽出'))}")
    print(f"   ↳ 🎯 [本體結構] core_object = {normalize_whitespace(ontology_profile.get('core_object', '未抽出'))}")
    print(f"   ↳ 🎯 [多視角展開] 成功釋放 {len(probe_bundle)} 組有效探針：")
    for i, item in enumerate(probe_bundle, 1):
        print(f"      - [{item['role']}] {item['statement']}")

    try:
        by_key = {}
        for k in DIMENSION_KEYS:
            if f"{k}_score" not in res or f"{k}_confidence" not in res:
                raise ValueError(f"缺少維度分數或置信度：{k}")
            by_key[k] = {
                "signed_score": res.get(f"{k}_score", 0),
                "confidence": res.get(f"{k}_confidence", 0),
                "reason": str(res.get(f"{k}_reason", "無"))
            }
    except Exception as e:
        print(f"⚠️ [維度破裂] LLM 遺失必要欄位！原因：{e}")
        by_key = {k: {"signed_score": 0, "confidence": 0, "reason": "全文量化失敗，啟動保底"} for k in DIMENSION_KEYS}

    profile = {
        "ontology_profile": {
            "topic_world": normalize_whitespace(ontology_profile.get("topic_world", "未抽出")),
            "core_object": normalize_whitespace(ontology_profile.get("core_object", "未抽出")),
            "mechanism_language": unique_list(ontology_profile.get("mechanism_language", []) if isinstance(ontology_profile.get("mechanism_language", []), list) else [], 20),
            "supporting_anchors": unique_list(ontology_profile.get("supporting_anchors", []) if isinstance(ontology_profile.get("supporting_anchors", []), list) else [], 20),
            "metaphor_anchors": unique_list(ontology_profile.get("metaphor_anchors", []) if isinstance(ontology_profile.get("metaphor_anchors", []), list) else [], 20),
            "forbidden_or_opposed": unique_list(ontology_profile.get("forbidden_or_opposed", []) if isinstance(ontology_profile.get("forbidden_or_opposed", []), list) else [], 20),
        },
        "retrieval_signature_en": retrieval_signature,
        "primary_statement": normalize_whitespace(primary_statement),
        "probe_bundle": probe_bundle,
        "valid_statements": [p["statement"] for p in probe_bundle],
        "implementation_signals": unique_list(res.get("implementation_signals", []) if isinstance(res.get("implementation_signals", []), list) else [], 20),
        "application_signals": unique_list(res.get("application_signals", []) if isinstance(res.get("application_signals", []), list) else [], 20),
        "academic_fingerprint": normalize_whitespace(res.get("academic_fingerprint", "預設紀錄")),
        "scores": {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS},
        "confidences": {k: enforce_confidence(by_key[k].get("confidence"), k) for k in DIMENSION_KEYS},
        "reasons": {k: normalize_whitespace(by_key[k].get("reason", "")) for k in DIMENSION_KEYS},
    }
    profile["hex_code"] = sign_to_binary(profile["scores"])
    return profile


def unique_probe_bundle(bundle, limit=12):
    seen = set()
    out = []
    for item in bundle:
        role = item.get("role", "")
        statement = normalize_statement(item.get("statement", ""))
        key = (role, statement.lower())
        if statement and key not in seen:
            seen.add(key)
            out.append({"role": role, "statement": statement})
        if len(out) >= limit:
            break
    return out


def fallback_statement_from_text(raw_text):
    match = re.search(r'(?:#+)\s*([A-Za-z0-9\-\s:]+)', raw_text)
    if not match:
        match = re.search(r'\(([A-Za-z0-9\-\s]{10,80})\)', raw_text)
    if not match:
        match = re.search(r'([A-Za-z0-9\-\s]{15,80})', raw_text)
    return normalize_statement(match.group(1)) if match else "核心論述提取失敗"


def repair_application_dimension_if_needed(raw_text, profile):
    # keep generic evidence repair, not topic hard-coded
    app_score = profile["scores"]["application"]
    evidence_count = len(profile["implementation_signals"]) + len(profile["application_signals"])
    if app_score > 0 or evidence_count < 3:
        return profile

    print("   ↳ 🛠️ [應用實相校正] 偵測到全文存在明確實作/輸出證據，但 D6 <= 0，啟動全文再判定...")
    sys_prompt = """
你是一台「應用實相矛盾校正器」。
只重新判斷 application 維度。
若全文已存在可執行流程、引擎、外部檢索、結構化輸出或可驗證資產，就不能把 application 判成純理論停留。
只能回傳 JSON：
{
  "application_score": 0,
  "application_confidence": 0,
  "application_reason": "中文短語"
}
""".strip()
    user_prompt = (
        f"【全文】\n{raw_text}\n\n"
        f"【原始 application 判定】score={app_score}, reason={profile['reasons']['application']}\n"
        f"【implementation_signals】{json.dumps(profile['implementation_signals'], ensure_ascii=False)}\n"
        f"【application_signals】{json.dumps(profile['application_signals'], ensure_ascii=False)}"
    )
    try:
        res = parse_llm_json(call_local_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ], json_mode=True, num_ctx=FULLTEXT_NUM_CTX))
        repaired_score = enforce_score(res.get("application_score", app_score), "application_score")
        repaired_conf = enforce_confidence(res.get("application_confidence", profile["confidences"]["application"]), "application_confidence")
        repaired_reason = normalize_whitespace(res.get("application_reason", profile["reasons"]["application"]))
        if repaired_score <= 0:
            repaired_score = max(35, app_score)
            repaired_conf = max(88, repaired_conf)
            repaired_reason = "全文已明確存在可執行流程與輸出資產，依應用實相定義校正為弱正值"
        profile["scores"]["application"] = repaired_score
        profile["confidences"]["application"] = repaired_conf
        profile["reasons"]["application"] = repaired_reason
        profile["hex_code"] = sign_to_binary(profile["scores"])
    except Exception as e:
        print(f"      ⚠️ 應用實相校正失敗，保留原值 ({e})")
    return profile


# ------------------------------------------------------------------------------
# Candidate ontology classification (generic, no hard-coded domain lexicon)
# ------------------------------------------------------------------------------

def classify_candidate_against_ontology(profile, probe_role, probe_statement, candidate):
    sys_prompt = """
你是一台「候選文獻本體對位審核器」。
你要判斷候選文獻和原文是否處於同一個主題世界，而不是只撞到相同機制語言或比喻語彙。
只能回傳 JSON。

relation_type 只能是以下四種之一：
- topic_aligned：主題世界對位，屬於可進正式背景池的真正鄰域。
- mechanism_only：只抓到方法/機制/形式語言，但主題世界不一致。
- metaphor_only：只抓到比喻/修辭/隱喻詞，不是同一個主題世界。
- topic_escape：已經漂移到另一個主題世界。

判斷原則：
1. topic_world 與 core_object 最重要。
2. mechanism_language 只能輔助，不得凌駕 topic_world。
3. metaphor_anchors 不能主導判定。
4. forbidden_or_opposed 若大量命中，傾向 topic_escape 或 mechanism_only。
5. 不要因為出現 tensor/field/state/wave 等字就自動認為 topic_aligned。

JSON 結構：
{
  "relation_type": "topic_aligned",
  "topic_world_fit": 0,
  "core_object_fit": 0,
  "mechanism_fit": 0,
  "metaphor_pull": 0,
  "escape_risk": 0,
  "decision_score": 0,
  "reason": "中文短語"
}
""".strip()

    payload = {
        "ontology_profile": profile["ontology_profile"],
        "retrieval_signature_en": profile["retrieval_signature_en"],
        "primary_statement": profile["primary_statement"],
        "probe_role": probe_role,
        "probe_statement": probe_statement,
        "candidate": {
            "title": candidate["title"],
            "abstract": candidate["abstract"]
        }
    }

    try:
        res = parse_llm_json(call_local_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ], json_mode=True, num_ctx=CLASSIFY_NUM_CTX))
        relation_type = normalize_whitespace(res.get("relation_type", "topic_escape")).lower()
        if relation_type not in {"topic_aligned", "mechanism_only", "metaphor_only", "topic_escape"}:
            relation_type = "topic_escape"
        return {
            "relation_type": relation_type,
            "topic_world_fit": clamp(int(round(float(res.get("topic_world_fit", 0)))), 0, 100),
            "core_object_fit": clamp(int(round(float(res.get("core_object_fit", 0)))), 0, 100),
            "mechanism_fit": clamp(int(round(float(res.get("mechanism_fit", 0)))), 0, 100),
            "metaphor_pull": clamp(int(round(float(res.get("metaphor_pull", 0)))), 0, 100),
            "escape_risk": clamp(int(round(float(res.get("escape_risk", 0)))), 0, 100),
            "decision_score": clamp(int(round(float(res.get("decision_score", 0)))), 0, 100),
            "reason": normalize_whitespace(res.get("reason", "無"))
        }
    except Exception as e:
        print(f"      ⚠️ 候選文獻本體審核失敗，預設為 topic_escape ({e})")
        return {
            "relation_type": "topic_escape",
            "topic_world_fit": 0,
            "core_object_fit": 0,
            "mechanism_fit": 0,
            "metaphor_pull": 0,
            "escape_risk": 100,
            "decision_score": 0,
            "reason": "本體審核失敗，預設外場逃逸"
        }


# ------------------------------------------------------------------------------
# Retrieval + ontology rerank
# ------------------------------------------------------------------------------

def multi_perspective_retrieval_and_rerank(profile):
    probe_bundle = profile.get("probe_bundle", [])
    if not probe_bundle:
        print("🌍 [階段 2] 核心論述失效，中斷 Crossref 打撈，系統將自然回歸無人區狀態。")
        return [], [], 0, [], []

    global_candidate_pool = []
    retrieval_logs = []
    raw_hits_count = 0
    shadow_hits_global = []
    escaped_hits_global = []

    signature_vec = get_text_vector(profile["retrieval_signature_en"] + " " + profile["primary_statement"])

    print(f"🌍 [階段 2 & 3] 啟動多視角打撈與本體主軸收斂 (共 {len(probe_bundle)} 組探針，最大搜索量 {len(probe_bundle) * RETRIEVAL_ROWS_PER_PROBE} 篇)...")

    for idx, probe in enumerate(probe_bundle):
        role = probe["role"]
        stmt = probe["statement"]
        print(f"   ↳ ⏳ [視角 {idx + 1}/{len(probe_bundle)} | {role}] 發射論述: {stmt}")
        stmt_vec = get_text_vector(stmt)

        url = (
            f"https://api.crossref.org/works?query={urllib.parse.quote(stmt)}"
            f"&select=DOI,title,abstract,author,issued&rows={RETRIEVAL_ROWS_PER_PROBE}"
        )
        try:
            response = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/57.0"}, timeout=20)
            if response.status_code == 429:
                time.sleep(5)
                response = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/57.0"}, timeout=20)
            response.raise_for_status()
        except Exception as e:
            print(f"      ⚠️ API 呼叫失敗 ({e})")
            retrieval_logs.append(f"* **視角 {idx + 1}** `[{role}] {stmt}`\n  * ⚠️ 打撈落空：Crossref API 呼叫失敗或超時")
            continue

        items = response.json().get("message", {}).get("items", [])
        raw_hits_count += len(items)
        if not items:
            retrieval_logs.append(f"* **視角 {idx + 1}** `[{role}] {stmt}`\n  * ⚠️ 打撈落空：Crossref 回傳 0 篇")
            print("      ⚠️ API 回傳 0 篇")
            continue

        scored_for_this_stmt = []
        for paper in items:
            title_list = paper.get("title")
            title = normalize_whitespace(title_list[0] if title_list else "Unknown")
            doi = str(paper.get("DOI", "Unknown")).strip()
            abs_text = clean_crossref_abstract(paper.get("abstract", ""))
            abs_source = "crossref_list" if abs_text else ""
            if not abs_text and doi and doi != "Unknown":
                ext_abs, ext_source = fetch_external_abstract(doi)
                if ext_abs:
                    abs_text = ext_abs
                    abs_source = ext_source

            if not abs_text:
                eval_text = title
                display_abs = "（此文獻於資料庫與外部補抓來源均無提供摘要，系統已降格為殘影，不參與正式背景場。）"
            else:
                eval_text = title + " " + abs_text
                display_abs = abs_text[:900]

            paper_vec = get_text_vector(eval_text)
            probe_sim = compute_dict_cosine(stmt_vec, paper_vec)
            signature_sim = compute_dict_cosine(signature_vec, paper_vec)
            combined_score = probe_sim * 0.55 + signature_sim * 0.45

            authors = paper.get("author", [])
            first_author = str(authors[0].get("family", "")).lower().strip() if authors else "unknown"
            try:
                year = int(paper.get("issued", {}).get("date-parts", [[0]])[0][0])
            except Exception:
                year = 0

            candidate = {
                "id": doi,
                "title": title,
                "abstract": display_abs,
                "author": first_author,
                "year": year,
                "probe_role": role,
                "source_statement": stmt,
                "probe_similarity": round(probe_sim, 4),
                "signature_similarity": round(signature_sim, 4),
                "similarity": round(combined_score, 4),
                "has_abs": bool(abs_text),
                "abstract_source": abs_source,
            }

            if abs_text and combined_score >= DOC_CAPTURE_THRESHOLD:
                ont = classify_candidate_against_ontology(profile, role, stmt, candidate)
                candidate.update(ont)
            else:
                candidate.update({
                    "relation_type": "metaphor_only" if abs_text else "topic_escape",
                    "topic_world_fit": 0,
                    "core_object_fit": 0,
                    "mechanism_fit": 0,
                    "metaphor_pull": 0,
                    "escape_risk": 100,
                    "decision_score": 0,
                    "reason": "摘要缺失或字面命中不足"
                })
            scored_for_this_stmt.append(candidate)

        scored_for_this_stmt.sort(key=lambda x: (x["similarity"], x.get("decision_score", 0)), reverse=True)

        top3_log = "  * 📊 **Top 3 綜合命中度**：\n"
        for i, c in enumerate(scored_for_this_stmt[:3]):
            top3_log += (
                f"    {i + 1}. `[{c['similarity']:.3f}]` {c['title'][:60]}... "
                f"(probe={c['probe_similarity']:.3f}, signature={c['signature_similarity']:.3f}, relation={c['relation_type']})\n"
            )

        effective_hits = [c for c in scored_for_this_stmt if c["similarity"] >= DOC_CAPTURE_THRESHOLD and c["has_abs"] and c["relation_type"] == "topic_aligned"]
        shadow_hits = [c for c in scored_for_this_stmt if c["similarity"] >= DOC_CAPTURE_THRESHOLD and (not c["has_abs"] or c["relation_type"] in {"mechanism_only", "metaphor_only"})]
        escaped_hits = [c for c in scored_for_this_stmt if c["similarity"] >= DOC_CAPTURE_THRESHOLD and c["has_abs"] and c["relation_type"] == "topic_escape"]

        if effective_hits:
            best = effective_hits[0]
            status_log = (
                f"  * 🎯 **正式背景捕獲（topic_aligned）**：[{best['title']}](https://doi.org/{best['id']}) "
                f"(Score: `{best['similarity']:.3f}` ｜ ontology `{best['decision_score']}` ｜ {best['reason']})"
            )
            print(f"      ✅ 視角正式捕獲: {best['title'][:40]}... ({best['similarity']:.3f}, ontology={best['decision_score']})")
        else:
            status_log = "  * ⚠️ **正式背景落空**：此視角沒有任何 topic_aligned 的摘要可驗證文獻。"
            print("      ⚠️ 視角正式背景落空：無 topic_aligned 文獻。")

        if shadow_hits:
            best_shadow = shadow_hits[0]
            shadow_log = (
                f"  * 🪧 **候補/殘影**：[{best_shadow['title']}](https://doi.org/{best_shadow['id']}) "
                f"(Score: `{best_shadow['similarity']:.3f}` ｜ relation `{best_shadow['relation_type']}` ｜ {best_shadow['reason']})"
            )
            shadow_hits_global.extend(shadow_hits[:3])
        else:
            shadow_log = "  * 🪧 **候補/殘影**：無"

        if escaped_hits:
            best_escape = escaped_hits[0]
            escape_log = (
                f"  * 🚫 **Topic Escape**：[{best_escape['title']}](https://doi.org/{best_escape['id']}) "
                f"(Score: `{best_escape['similarity']:.3f}` ｜ escape `{best_escape['escape_risk']}` ｜ {best_escape['reason']})"
            )
            escaped_hits_global.extend(escaped_hits[:3])
        else:
            escape_log = "  * 🚫 **Topic Escape**：無"

        retrieval_logs.append(f"* **視角 {idx + 1}** `[{role}] {stmt}`\n{top3_log}{status_log}\n{shadow_log}\n{escape_log}")
        global_candidate_pool.extend(effective_hits)

    print(f"🌍 [全域顯化] 正式背景池共有 {len(global_candidate_pool)} 篇 topic_aligned 文獻，啟動全域聚合排序...")
    if not global_candidate_pool:
        return [], retrieval_logs, raw_hits_count, shadow_hits_global, escaped_hits_global

    aggregated = {}
    for c in global_candidate_pool:
        key = c["id"] if c["id"] != "Unknown" else c["title"].lower()
        if key not in aggregated:
            aggregated[key] = {
                "id": c["id"],
                "title": c["title"],
                "abstract": c["abstract"],
                "author": c["author"],
                "year": c["year"],
                "has_abs": True,
                "abstract_source": c["abstract_source"],
                "max_similarity": c["similarity"],
                "sum_similarity": c["similarity"],
                "hit_count": 1,
                "source_statements": {c["source_statement"]},
                "max_probe_similarity": c["probe_similarity"],
                "max_signature_similarity": c["signature_similarity"],
                "max_decision_score": c.get("decision_score", 0),
                "best_reason": c.get("reason", "")
            }
        else:
            agg = aggregated[key]
            agg["max_similarity"] = max(agg["max_similarity"], c["similarity"])
            agg["sum_similarity"] += c["similarity"]
            agg["hit_count"] += 1
            agg["source_statements"].add(c["source_statement"])
            agg["max_probe_similarity"] = max(agg["max_probe_similarity"], c["probe_similarity"])
            agg["max_signature_similarity"] = max(agg["max_signature_similarity"], c["signature_similarity"])
            agg["max_decision_score"] = max(agg["max_decision_score"], c.get("decision_score", 0))
            if c.get("decision_score", 0) >= agg["max_decision_score"]:
                agg["best_reason"] = c.get("reason", agg["best_reason"])

    merged_candidates = []
    for agg in aggregated.values():
        source_count = len(agg["source_statements"])
        avg_similarity = agg["sum_similarity"] / agg["hit_count"]
        global_score = (
            agg["max_similarity"] * 0.35
            + avg_similarity * 0.15
            + agg["max_signature_similarity"] * 0.10
            + agg["max_probe_similarity"] * 0.10
            + (agg["max_decision_score"] / 100.0) * 0.25
            + min(source_count, 4) * 0.03
            + min(agg["hit_count"], 4) * 0.02
        )
        merged_candidates.append({
            "id": agg["id"],
            "title": agg["title"],
            "abstract": agg["abstract"],
            "author": agg["author"],
            "year": agg["year"],
            "has_abs": True,
            "abstract_source": agg["abstract_source"],
            "similarity": round(agg["max_similarity"], 4),
            "avg_similarity": round(avg_similarity, 4),
            "hit_count": agg["hit_count"],
            "source_count": source_count,
            "source_statements": sorted(list(agg["source_statements"])),
            "max_probe_similarity": round(agg["max_probe_similarity"], 4),
            "max_signature_similarity": round(agg["max_signature_similarity"], 4),
            "ontology_score": agg["max_decision_score"],
            "best_reason": agg["best_reason"],
            "global_score": round(global_score, 4),
        })

    merged_candidates.sort(key=lambda x: (x["global_score"], x["ontology_score"], x["max_signature_similarity"], x["source_count"]), reverse=True)

    final_papers, seen_dois, seen_titles = [], set(), []
    for candidate in merged_candidates:
        if candidate["id"] in seen_dois:
            continue
        if any(is_similar_title(candidate["title"], st) for st in seen_titles):
            continue
        final_papers.append(candidate)
        seen_dois.add(candidate["id"])
        seen_titles.append(candidate["title"])
        if len(final_papers) >= MAX_GLOBAL_FINAL:
            break

    print(f"🌍 全域收斂完成：從正式背景池中萃取出 {len(final_papers)} 篇本體對位文獻，準備進入六維量化...")
    return final_papers, retrieval_logs, raw_hits_count, shadow_hits_global, escaped_hits_global


# ------------------------------------------------------------------------------
# Background scoring
# ------------------------------------------------------------------------------

def evaluate_background_papers(final_papers, core_statement, retrieval_signature_en, ontology_profile):
    if not final_papers:
        return {"papers": [], "batch_log": "無正式背景文獻。"}
    print(f"📚 [階段 4] 啟動「切片吞吐」模式，逐篇量化 {len(final_papers)} 篇背景文獻以保護 VRAM...")
    scored_papers = []
    for i, paper in enumerate(final_papers):
        print(f"   ↳ ⏳ [切片吞吐 {i + 1}/{len(final_papers)}] 正在消化: {paper['title'][:30]}...")
        sys_prompt = f"""
觀測原點 primary_statement："{core_statement}"
全文檢索簽名 retrieval_signature_en："{retrieval_signature_en}"
ontology_profile：{json.dumps(ontology_profile, ensure_ascii=False)}

請量化以下這【1】篇文獻。維度定義：
{build_dimensions_prompt()}

回傳扁平化 JSON 格式：
{{
  "note": "短中文",
  "value_intent_score": 25,
  "governance_score": 10,
  "cognition_score": 40,
  "architecture_score": 35,
  "expansion_score": 20,
  "application_score": -15
}}
""".strip()
        user_prompt = f"【待測背景文獻】\n{json.dumps(paper, ensure_ascii=False)}"
        try:
            res = parse_llm_json(call_local_llm([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ], json_mode=True, num_ctx=BACKGROUND_NUM_CTX))
            by_key = {k: {"signed_score": res.get(f"{k}_score", 0)} for k in DIMENSION_KEYS}
            scored_papers.append({
                "id": paper["id"],
                "title": paper["title"],
                "note": normalize_whitespace(res.get("note", paper.get("best_reason", ""))),
                "scores": {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS},
                "has_abs": True,
                "source_count": paper.get("source_count", 1),
                "abstract_source": paper.get("abstract_source", "")
            })
        except Exception as e:
            print(f"      ⚠️ 該篇文獻消化失敗，觸發動態卸力，直接略過 ({e})")
            continue
        time.sleep(1)
    return {"papers": scored_papers, "batch_log": f"系統採用切片吞吐模式，成功量化正式背景池中的 {len(scored_papers)}/{len(final_papers)} 篇文獻。"}


# ------------------------------------------------------------------------------
# Vector interference + formatting
# ------------------------------------------------------------------------------

def aggregate_background(scored_papers):
    mean_scores, peak_scores, peak_papers = {}, {}, {}
    for key in DIMENSION_KEYS:
        vals = [(p["scores"][key], p) for p in scored_papers]
        mean_scores[key] = round(sum(v for v, _ in vals) / len(vals), 1)
        peak_val, peak_paper = max(vals, key=lambda x: x[0])
        peak_scores[key] = peak_val
        peak_papers[key] = peak_paper
    background_hex = sign_to_binary({k: mean_scores[k] for k in DIMENSION_KEYS})
    return mean_scores, peak_scores, peak_papers, background_hex


def build_vector_logs(user_profile, scored_papers):
    user_scores = user_profile["scores"]
    mean_scores, peak_scores, peak_papers, background_hex = aggregate_background(scored_papers)
    user_vec = [user_scores[k] for k in DIMENSION_KEYS]
    bg_vec = [mean_scores[k] for k in DIMENSION_KEYS]
    cos_val = compute_dict_cosine(dict(enumerate(user_vec)), dict(enumerate(bg_vec)))
    angle = round(angle_from_cosine(cos_val), 1)
    avg_u = sum(user_vec) / 6
    avg_b = sum(bg_vec) / 6
    mean_diff_global = abs(avg_u - avg_b)
    global_proximity = round(max(0.0, 100.0 - ((angle / 1.8) * 0.4 + mean_diff_global * 0.6)), 1)

    if angle < 30:
        base_rel = "高度同向"
    elif angle < 60:
        base_rel = "中度同向"
    elif angle < 90:
        base_rel = "弱同向"
    elif angle == 90:
        base_rel = "正交"
    elif angle < 120:
        base_rel = "弱反向"
    else:
        base_rel = "明顯反向"

    if angle < 90:
        if avg_u > avg_b + 40:
            global_relation = f"{base_rel}（全域能勢大幅突破）"
        elif avg_b > avg_u + 40:
            global_relation = f"{base_rel}（全域能勢大幅覆蓋）"
        else:
            global_relation = f"{base_rel}（能勢共振相近）"
    else:
        global_relation = base_rel

    vector_logs = []
    for key in DIMENSION_KEYS:
        u = user_scores[key]
        b = mean_scores[key]
        peak = peak_scores[key]
        peak_paper = peak_papers[key]
        proximity = proximity_from_scores(u, b)
        relation = classify_relation(u, b)
        diff_mean = round(u - b, 1)
        diff_peak = round(u - peak, 1)
        peak_compare = "背景能勢覆蓋" if abs(peak) > abs(u) else "本體能勢突破"
        vector_logs.append({
            "key": key, "label": dim_label(key), "user_score": u,
            "background_mean": b, "background_peak": peak,
            "peak_title": compact_title(peak_paper["title"]),
            "relation": relation, "proximity": proximity,
            "diff_mean": diff_mean, "diff_peak": diff_peak,
            "peak_compare": peak_compare,
        })

    return {
        "background_hex": background_hex,
        "mean_scores": mean_scores,
        "peak_scores": peak_scores,
        "peak_papers": peak_papers,
        "global_angle": angle,
        "global_cosine": round(cos_val, 4),
        "global_proximity": global_proximity,
        "global_relation": global_relation,
        "vector_logs": vector_logs,
    }


def format_user_dimension_logs(user_profile):
    logs = []
    for key in DIMENSION_KEYS:
        logs.append(
            f"* **{dim_label(key)}**：`{user_profile['scores'][key]:+d}` / 100 ｜ **{signed_score_to_side(user_profile['scores'][key])}** ｜ "
            f"置信度 `{user_profile['confidences'][key]}` ｜ 觀測判定：{user_profile['reasons'][key]}"
        )
    return logs


def format_vector_logs(vector_data):
    logs = []
    for item in vector_data["vector_logs"]:
        logs.append(
            f"* **{item['label']}**：本體 `{item['user_score']:+d}` ｜ 背景均值 `{item['background_mean']:+.1f}` ｜ "
            f"背景峰值 `{item['background_peak']:+d}`（{item['peak_title']}） ｜ 方向 `{item['relation']}` ｜ "
            f"相近度 `{item['proximity']}` / 100 ｜ 均值差 `{item['diff_mean']:+.1f}` ｜ 峰值差 `{item['diff_peak']:+.1f}` ｜ {item['peak_compare']}"
        )
    return logs


def format_reference_records(scored_papers):
    rows = []
    for p in scored_papers:
        doi_link = f"https://doi.org/{p['id']}" if p["id"] != "Unknown" else "#"
        hit_marker = f" ｜多視角命中 `{p.get('source_count', 1)}`" if p.get("source_count", 1) > 1 else ""
        src_marker = f" ｜摘要源 `{p.get('abstract_source', '')}`" if p.get("abstract_source", "") else ""
        rows.append(f"- [DOI 連結]({doi_link}) **{p['title']}**{hit_marker}{src_marker}")
    return rows if rows else ["- 無"]


def format_shadow_records(shadow_hits):
    rows = []
    for s in shadow_hits[:MAX_SHADOW_LOG]:
        doi_link = f"https://doi.org/{s['id']}" if s["id"] != "Unknown" else "#"
        rows.append(
            f"- [DOI 連結]({doi_link}) **{s['title']}** ｜relation `{s['relation_type']}` ｜"
            f" score `{s['similarity']}` ｜ probe `{s['probe_similarity']}` ｜ signature `{s['signature_similarity']}`"
        )
    return rows if rows else ["- 無"]


def format_escape_records(escaped_hits):
    rows = []
    for s in escaped_hits[:MAX_SHADOW_LOG]:
        doi_link = f"https://doi.org/{s['id']}" if s["id"] != "Unknown" else "#"
        rows.append(
            f"- [DOI 連結]({doi_link}) **{s['title']}** ｜escape `{s['escape_risk']}` ｜"
            f" score `{s['similarity']}` ｜ {s['reason']}"
        )
    return rows if rows else ["- 無"]


def generate_summary(raw_text, global_relation, global_angle, global_proximity):
    prompt = f"""
本理論在外部背景場中的整體關係為：{global_relation}。
整體相位角：約 {global_angle} 度。
整體語意相近度：約 {global_proximity} / 100。

請根據下文全文，撰寫 180-240 字中文理論導讀。第一句必須以「本理論架構...」開頭。客觀不神話化。
""".strip()
    try:
        res_text = call_local_llm([
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_text}
        ], temperature=0.2, num_ctx=SUMMARY_NUM_CTX)
        return zhconv.convert(res_text.strip(), "zh-tw")
    except Exception:
        return f"本理論架構目前與背景場的整體關係為{global_relation}，相位角約為 {global_angle} 度，語意相近度約為 {global_proximity} / 100。由於生成階段發生偏移，系統暫以保底敘述輸出。"


# ------------------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------------------

def process_avh_manifestation(source_path):
    print(f"\n🌊 [波包掃描] 實體源碼：{os.path.basename(source_path)}")
    try:
        with open(source_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        if len(raw_text.strip()) < 100:
            return None

        user_profile = evaluate_user_profile(raw_text)
        user_profile = repair_application_dimension_if_needed(raw_text, user_profile)
        user_hex = user_profile["hex_code"]
        state_info = MANIFEST["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})

        try:
            final_papers, retrieval_logs, raw_hits_count, shadow_hits, escaped_hits = multi_perspective_retrieval_and_rerank(user_profile)
        except Exception as e:
            final_papers, retrieval_logs, raw_hits_count, shadow_hits, escaped_hits = [], [f"打撈或收斂失敗（{e}）"], 0, [], []

        scored_background = evaluate_background_papers(
            final_papers,
            user_profile["primary_statement"],
            user_profile["retrieval_signature_en"],
            user_profile["ontology_profile"]
        ) if final_papers else {"papers": []}

        topic_escape_dominant = len(escaped_hits) > len(final_papers) and len(escaped_hits) >= 3

        if not scored_background["papers"]:
            if topic_escape_dominant:
                baseline_status = "Topic Escape（檢索語言失真：外場被異主題世界接管）"
                background_batch_log = "多數高分候選被本體審核判定為 topic_escape。系統拒絕把機制撞詞或異主題世界文獻誤當正式背景場。"
            elif shadow_hits:
                baseline_status = "Void（僅有候補/殘影：無可驗證本體母體）"
                background_batch_log = "本輪存在若干候補/殘影節點，但本體審核後無法形成正式背景場。系統誠實回到無人區。"
            elif final_papers:
                baseline_status = f"Void（觀測破裂：{len(final_papers)} 篇背景文獻量化全數失敗）"
                background_batch_log = "打撈已保留文獻，但背景文獻逐篇量化時 LLM 發生崩潰，無法形成有效母體。"
            else:
                baseline_status = "Void（無人區：外部場域尚不足以形成可測量母體）"
                background_batch_log = "最終保留文獻為 0，系統判定當前外部場域不足以構成可測量背景母體。"
            background_hex = "000000"
            vector_logs = ["* **背景向量量化**：無人區狀態、topic escape 或量化失敗，暫無穩定背景向量可供干涉比較。"]
            global_angle = "無定義（Void）"
            global_cosine = "N/A"
            global_proximity = "N/A"
            global_relation = "無人區"
            summary = "本理論架構目前未能形成穩定正式背景場；外部鄰近文獻要嘛不足、要嘛發生主題逃逸，因此與現有學界的方向關係暫時不可定義。"
        else:
            baseline_status = f"Background Field Established（本體對位背景場：{len(final_papers)} 鄰域節點）" if len(final_papers) >= MAX_GLOBAL_FINAL else f"Sparse Reference Field（本體對位鄰域：{len(final_papers)} 節點）"
            background_batch_log = scored_background["batch_log"]
            vector_data = build_vector_logs(user_profile, scored_background["papers"])
            background_hex = vector_data["background_hex"]
            vector_logs = format_vector_logs(vector_data)
            global_angle = f"{vector_data['global_angle']} 度"
            global_cosine = vector_data["global_cosine"]
            global_proximity = vector_data["global_proximity"]
            global_relation = vector_data["global_relation"]
            summary = generate_summary(raw_text, vector_data["global_relation"], vector_data["global_angle"], vector_data["global_proximity"])

        return {
            "user_hex": user_hex,
            "baseline_hex": background_hex,
            "state_name": state_info["name"],
            "state_desc": state_info["desc"],
            "summary": summary,
            "full_text": raw_text,
            "meta_data": {
                "primary_statement": user_profile["primary_statement"],
                "retrieval_signature_en": user_profile["retrieval_signature_en"],
                "probe_bundle": user_profile["probe_bundle"],
                "valid_statements": user_profile["valid_statements"],
                "ontology_profile": user_profile["ontology_profile"],
                "implementation_signals": user_profile["implementation_signals"],
                "application_signals": user_profile["application_signals"],
                "academic_fingerprint": user_profile["academic_fingerprint"],
                "user_dimension_logs": format_user_dimension_logs(user_profile),
                "raw_hits": raw_hits_count,
                "final_hits": len(final_papers) if final_papers else 0,
                "shadow_hits_count": len(shadow_hits),
                "escape_hits_count": len(escaped_hits),
                "retrieval_logs": retrieval_logs,
                "shadow_records": format_shadow_records(shadow_hits),
                "escape_records": format_escape_records(escaped_hits),
                "background_batch_log": background_batch_log,
                "paper_records": format_reference_records(scored_background["papers"]),
                "vector_logs": vector_logs,
                "baseline_status": baseline_status,
                "global_angle": global_angle,
                "global_cosine": global_cosine,
                "global_proximity": global_proximity,
                "global_relation": global_relation,
                "llm_model": OLLAMA_MODEL_NAME,
            }
        }
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        return None


# ------------------------------------------------------------------------------
# Output layers
# ------------------------------------------------------------------------------

def generate_trajectory_log(target_file, data):
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")
    meta = data["meta_data"]
    user_logs_text = "\n\n".join(meta["user_dimension_logs"])
    retrieval_text = "\n".join(meta["retrieval_logs"])
    vector_logs_text = "\n\n".join(meta["vector_logs"])
    papers_text = "\n".join(meta["paper_records"])
    shadow_text = "\n".join(meta["shadow_records"])
    escape_text = "\n".join(meta["escape_records"])
    probe_str = " ｜ ".join([f"[{p['role']}] {p['statement']}" for p in meta["probe_bundle"]]) if meta["probe_bundle"] else "未釋放"
    impl_str = " ｜ ".join(meta["implementation_signals"]) if meta["implementation_signals"] else "未萃取"
    app_sig_str = " ｜ ".join(meta["application_signals"]) if meta["application_signals"] else "未萃取"
    ontology = meta["ontology_profile"]
    ontology_lines = (
        f"* **topic_world**：`{ontology.get('topic_world', '未抽出')}`\n"
        f"* **core_object**：`{ontology.get('core_object', '未抽出')}`\n"
        f"* **supporting_anchors**：`{' ｜ '.join(ontology.get('supporting_anchors', [])) if ontology.get('supporting_anchors') else '未抽出'}`\n"
        f"* **mechanism_language**：`{' ｜ '.join(ontology.get('mechanism_language', [])) if ontology.get('mechanism_language') else '未抽出'}`\n"
        f"* **metaphor_anchors**：`{' ｜ '.join(ontology.get('metaphor_anchors', [])) if ontology.get('metaphor_anchors') else '未抽出'}`\n"
        f"* **forbidden_or_opposed**：`{' ｜ '.join(ontology.get('forbidden_or_opposed', [])) if ontology.get('forbidden_or_opposed') else '未抽出'}`"
    )

    return (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳（{tz_name}）**：`{timestamp}`\n"
        f"* **高維算力引擎（本地純淨版）**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 絕對本體觀測（Absolute Ontology）\n"
        f"* 🛡️ **本體論絕對指紋（Ontology Hex）**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"  * 📜 **演化實相（State Manifest）**：_{data['state_desc']}_\n"
        f"* **絕對核心論述（Primary Statement）**：`{meta['primary_statement']}`\n"
        f"* **全文檢索簽名（Retrieval Signature）**：`{meta['retrieval_signature_en']}`\n"
        f"* **多視角外發探針（Probe Set）**：`{probe_str}`\n"
        f"* **實作證據（Implementation Signals）**：`{impl_str}`\n"
        f"* **應用證據（Application Signals）**：`{app_sig_str}`\n\n"
        f"**Ontology Profile**：\n"
        f"{ontology_lines}\n\n"
        f"**學術指紋（Academic Fingerprint）**：\n"
        f"> {meta['academic_fingerprint']}\n\n"
        f"**詳細本體量化儀表板（Ontology Quantification Dashboard）**：\n\n"
        f"{user_logs_text}\n\n"
        f"---\n"
        f"### 2. 🎣 背景能勢打撈（Background Field Retrieval）\n"
        f"* **場域建構狀態（Field Status）**：`{meta['baseline_status']}` （多視角搜索共 {meta['raw_hits']} 篇 → 正式背景收斂至 {meta['final_hits']} 篇；候補/殘影 {meta['shadow_hits_count']} 筆；topic escape {meta['escape_hits_count']} 筆）\n"
        f"* **光譜透析多視角打撈日誌（Spectrum Dialysis Retrieval Log）**：\n"
        f"{retrieval_text}\n\n"
        f"* **候補/殘影區（Not formal background）**：\n"
        f"{shadow_text}\n\n"
        f"* **Topic Escape 區（Escaped topic-world hits）**：\n"
        f"{escape_text}\n\n"
        f"* **背景批次量化摘要（Batch Quantification Log）**：_{meta['background_batch_log']}_\n"
        f"* **全域收斂之正式能勢節點（Ontology-aligned Global Reference Nodes）**：\n"
        f"{papers_text}\n\n"
        f"---\n"
        f"### 3. 📐 向量干涉量化（Quantified Vector Interference）\n"
        f"* **背景絕對指紋（Background Hex）**：`[{data['baseline_hex']}]`\n"
        f"* **整體場域關係（Global Relation）**：**{meta['global_relation']}**\n"
        f"* **整體相位角（Global Angle）**：`{meta['global_angle']}`\n"
        f"* **全域餘弦相似（Global Cosine Similarity）**：`{meta['global_cosine']}`\n"
        f"* **整體語意相近度（Global Semantic Proximity）**：`{meta['global_proximity']}` / 100\n"
        f"* **量化公式（Quantification Rule）**：`Per-dimension proximity = 100 - |U - B| / 2; Global proximity = 100 - ((Angle / 1.8) * 0.4 + Global_Mean_Diff * 0.6)`\n\n"
        f"**維度向量干涉儀表板（Per-Dimension Vector Dashboard）**：\n\n"
        f"{vector_logs_text}\n\n"
        f"---\n"
        f"### 4. 🧾 系統導讀摘要（System Interpretation）\n"
        f"> {data['summary']}\n\n"
        f"---\n"
    )


def export_wordpress_html(basename, data):
    safe_full_text = html.escape(data["full_text"]).replace("\n", "<br>")
    safe_summary = html.escape(data["summary"])
    meta = data["meta_data"]
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp_str = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "  <div class=\"avh-content\">\n"
        f"    {safe_full_text}\n"
        "  </div>\n"
        "  <hr>\n"
        "  <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "    <h3>📡 學術價值全像儀（AVH）主權算力認證</h3>\n"
        f"    <p><strong>絕對核心論述：</strong>{html.escape(meta['primary_statement'])}</p>\n"
        f"    <p><strong>全文檢索簽名：</strong>{html.escape(meta['retrieval_signature_en'])}</p>\n"
        f"    <p><strong>本體狀態：</strong>[ {html.escape(data['user_hex'])} ] - {html.escape(data['state_name'])}</p>\n"
        f"    <p><em>「{html.escape(data['state_desc'])}」</em></p>\n"
        f"    <p><strong>背景狀態：</strong>[ {html.escape(data['baseline_hex'])} ]</p>\n"
        f"    <p><strong>整體場域關係：</strong>{html.escape(str(meta['global_relation']))}</p>\n"
        f"    <p><strong>整體相位角：</strong>{html.escape(str(meta['global_angle']))}</p>\n"
        f"    <p><strong>整體語意相近度：</strong>{html.escape(str(meta['global_proximity']))} / 100</p>\n"
        f"    <p><strong>候補/殘影數：</strong>{meta['shadow_hits_count']} ｜ <strong>Topic Escape 數：</strong>{meta['escape_hits_count']}</p>\n"
        f"    <p><strong>理論導讀摘要：</strong><br>{safe_summary}</p>\n"
        f"    <p>物理時間戳：{timestamp_str}</p>\n"
        "  </div>\n"
        "</div>\n"
    )
    with open(os.path.join(BASE_DIR, f"WP_Ready_{basename}.html"), "w", encoding="utf-8") as f:
        f.write(html_output)


def export_latex(basename, data):
    safe_text = markdown_to_latex(data["full_text"])
    meta = data["meta_data"]
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp_str = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        f"\\title{{{simple_escape(basename)}}}\n"
        "\\author{Alaric Kuo}\n"
        f"\\date{{{timestamp_str}}}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        f"核心論述：{simple_escape(meta['primary_statement'])}\n\n"
        f"全文檢索簽名：{simple_escape(meta['retrieval_signature_en'])}\n\n"
        f"本體狀態：[{data['user_hex']}] {simple_escape(data['state_name'])}\n\n"
        f"演化實相：{simple_escape(data['state_desc'])}\n\n"
        f"背景狀態：[{data['baseline_hex']}]\n\n"
        f"整體場域關係：{simple_escape(str(meta['global_relation']))}\n\n"
        f"整體相位角：{simple_escape(str(meta['global_angle']))}\n\n"
        f"整體語意相近度：{simple_escape(str(meta['global_proximity']))}/100\n\n"
        f"候補/殘影數：{meta['shadow_hits_count']}；Topic Escape 數：{meta['escape_hits_count']}\n"
        "\\end{abstract}\n\n"
        f"{safe_text}\n\n"
        "\\end{document}\n"
    )
    with open(os.path.join(BASE_DIR, f"{basename}_Archive.tex"), "w", encoding="utf-8") as f:
        f.write(tex_output)


def ensure_git_identity():
    try:
        name = subprocess.run(["git", "config", "--get", "user.name"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
        email = subprocess.run(["git", "config", "--get", "user.email"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
        if not name:
            subprocess.run(["git", "config", "user.name", "AVH Local Bot"], check=False, cwd=BASE_DIR)
        if not email:
            subprocess.run(["git", "config", "user.email", "avh-local-bot@example.com"], check=False, cwd=BASE_DIR)
    except Exception:
        pass


def run_git_automation():
    print("\n🚀 [本地自動化] 正在推送到 GitHub...")
    try:
        ensure_git_identity()
        subprocess.run(["git", "add", "."], check=False, cwd=BASE_DIR)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False, cwd=BASE_DIR).returncode == 1:
            now = datetime.now().astimezone()
            tz_name = now.tzname() or "CST"
            commit_msg = f"🌌 自動顯化：本地算力推演定錨 ({now.strftime('%Y-%m-%d %H:%M')} {tz_name})"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "push"], check=True, cwd=BASE_DIR)
            print("✅ 推送完成！歷史已成功定錨。")
        else:
            print("ℹ️ 觀測結果無變動。")
    except Exception as e:
        print(f"❌ Git 同步失敗: {e}")


if __name__ == "__main__":
    md_files = [f for f in glob.glob(os.path.join(BASE_DIR, "*.md")) if os.path.basename(f).lower() != "avh_observation_log.md"]
    if not md_files:
        print("ℹ️ 未找到任何待測 Markdown 來源檔。")
        sys.exit(0)

    log_path = os.path.join(BASE_DIR, "AVH_OBSERVATION_LOG.md")
    success_count = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V57.0 本體主軸守門版\n---\n")
        for i, source in enumerate(md_files):
            print(f"\n{'=' * 60}")
            print(f"🚀 [物理觀測啟動] 處理進度 {i + 1}/{len(md_files)}: {os.path.basename(source)}")
            print(f"{'=' * 60}")
            try:
                data = process_avh_manifestation(source)
                if data:
                    success_count += 1
                    log_file.write(generate_trajectory_log(os.path.basename(source), data))
                    basename = os.path.splitext(os.path.basename(source))[0]
                    export_wordpress_html(basename, data)
                    export_latex(basename, data)
                print("\n❄️ [物理散熱] 實體質量處理完畢，強制進入 5 秒冷卻期，釋放 GPU VRAM 壓力...")
                time.sleep(5)
            except Exception as e:
                print(f"❌ [系統級崩潰] 處理 {os.path.basename(source)} 時發生致命錯誤: {e}")
                print("❄️ [物理保護] 異常中止，啟動 10 秒強制散熱，避免連續熱當機...")
                time.sleep(10)
    if success_count > 0:
        run_git_automation()
    else:
        print("❌ 無檔案成功完成處理。")
