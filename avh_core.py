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
import random
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V35.4 診斷穩定版 - GitHub Models Ping / JSON 保險 / 長退避)
# ==============================================================================

LLM_MODEL_NAME = "openai/gpt-4o"

DIMENSIONS = [
    {"key": "value_intent", "zh": "價值意圖", "en": "Value Intent"},
    {"key": "governance", "zh": "治理維度", "en": "Governance"},
    {"key": "cognition", "zh": "認知深度", "en": "Cognition"},
    {"key": "architecture", "zh": "描述架構", "en": "Architecture"},
    {"key": "expansion", "zh": "擴張潛力", "en": "Expansion"},
    {"key": "application", "zh": "應用實相", "en": "Application"},
]

DIMENSION_KEYS = [d["key"] for d in DIMENSIONS]
DIMENSION_META = {d["key"]: d for d in DIMENSIONS}

DIAG_STATE = {
    "status": "NOT_RUN",
    "message": "",
    "text_ping": "NOT_RUN",
    "json_ping": "NOT_RUN",
}

print(f"🧠 [載入觀測核心] 啟動 V35.4 診斷穩定版 ({LLM_MODEL_NAME})...")


def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("遺失 GITHUB_TOKEN，無法啟動算力。")
    return OpenAI(base_url="https://models.github.ai/inference", api_key=token)


def ensure_json_keyword(messages):
    """GitHub Models 在 json_object 模式下，messages 內最好明示 json。"""
    for m in messages:
        content = str(m.get("content", ""))
        if "json" in content.lower():
            return messages

    patched = list(messages)
    patched.insert(0, {
        "role": "system",
        "content": "Return valid json only. The response must be a single json object."
    })
    return patched


def classify_model_error(e):
    text = str(e).lower()

    if "too many requests" in text or "429" in text or "scraping github" in text or "rate limit" in text:
        return "RATE_LIMIT"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "AUTH"
    if "json_object" in text or "must contain the word 'json'" in text or 'must contain the word "json"' in text:
        return "JSON_MODE"
    if "timeout" in text:
        return "TIMEOUT"
    return "UNKNOWN"


def maybe_startup_jitter():
    """在 GitHub Actions 上先隨機退一步，避免一啟動就撞上共享熱場。"""
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        delay = random.randint(15, 45)
        print(f"⏳ [診斷前緩衝] 偵測到 GitHub Actions，共享熱場緩衝 {delay} 秒...")
        time.sleep(delay)


def run_github_model_diagnostics():
    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a diagnostic endpoint."},
                {"role": "user", "content": "Reply with OK only."}
            ]
        )
        text_reply = (response.choices[0].message.content or "").strip()
        DIAG_STATE["text_ping"] = f"OK: {text_reply[:80]}"
    except Exception as e:
        kind = classify_model_error(e)
        DIAG_STATE["status"] = f"FAIL_{kind}"
        DIAG_STATE["message"] = f"純文字 ping 失敗：{e}"
        DIAG_STATE["text_ping"] = f"FAIL: {e}"
        return DIAG_STATE

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=ensure_json_keyword([
                {"role": "system", "content": "Return valid json only."},
                {"role": "user", "content": 'Return this exact json object: {"status":"ok"}'}
            ])
        )
        json_reply = parse_llm_json(response.choices[0].message.content)
        DIAG_STATE["json_ping"] = f"OK: {json_reply}"
    except Exception as e:
        kind = classify_model_error(e)
        DIAG_STATE["status"] = f"FAIL_{kind}"
        DIAG_STATE["message"] = f"JSON ping 失敗：{e}"
        DIAG_STATE["json_ping"] = f"FAIL: {e}"
        return DIAG_STATE

    DIAG_STATE["status"] = "OK"
    DIAG_STATE["message"] = "GitHub Models 純文字與 JSON ping 均通過。"
    return DIAG_STATE


def call_llm_with_retry(client, messages, temperature=0.0, max_retries=4, json_mode=True):
    last_error = None

    for attempt in range(max_retries):
        try:
            effective_messages = ensure_json_keyword(messages) if json_mode else messages

            kwargs = {
                "messages": effective_messages,
                "model": LLM_MODEL_NAME,
                "temperature": temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            return client.chat.completions.create(**kwargs)

        except Exception as e:
            last_error = e
            kind = classify_model_error(e)

            if kind == "RATE_LIMIT":
                wait_schedule = [20, 40, 80, 160]
                wait_time = wait_schedule[min(attempt, len(wait_schedule) - 1)]
            else:
                wait_time = 2 ** attempt

            print(f"⚠️ 雲端連線異常 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試... [{e}]")

            if attempt < max_retries - 1:
                time.sleep(wait_time)

    raise ConnectionError(f"雲端算力請求超時或阻擋 ({last_error})")


def parse_llm_json(response_text):
    if response_text is None:
        raise ValueError("LLM 未回傳任何內容。")

    text = response_text.strip()

    fence = chr(96) * 3
    pattern = fence + r"(?:json)?\s*(.*?)\s*" + fence
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("找不到 JSON 起始符號 '{'")

    depth = 0
    in_string = False
    escape = False
    end_idx = -1

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
        if ch == "\\" and not escape:
            escape = True
        else:
            escape = False

    if end_idx == -1:
        raise ValueError("找不到完整 JSON 結尾。")

    clean_json = text[start_idx:end_idx + 1]

    try:
        return json.loads(clean_json, strict=False)
    except Exception:
        clean_json = re.sub(r"(?<!\\)\n", " ", clean_json)
        return json.loads(clean_json, strict=False)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_crossref_abstract(raw_abstract):
    if not raw_abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    text = html.unescape(text)
    return normalize_whitespace(text)


def clamp(value, low, high):
    return max(low, min(high, value))


def dim_label(key):
    meta = DIMENSION_META[key]
    return f"{meta['zh']}（{meta['en']}）"


def sign_to_binary(scores_by_key):
    return "".join("1" if scores_by_key[k] > 0 else "0" for k in DIMENSION_KEYS)


def signed_score_to_side(score):
    return "離群突破（sin）" if score > 0 else "合群守成（cos）"


def enforce_score(value, field_name):
    try:
        score = int(round(float(value)))
    except Exception:
        raise ValueError(f"{field_name} 分數無法解析：{value}")
    return clamp(score, -100, 100)


def enforce_confidence(value, field_name):
    try:
        conf = int(round(float(value)))
    except Exception:
        raise ValueError(f"{field_name} 置信度無法解析：{value}")
    return clamp(conf, 0, 100)


def validate_dimension_entries(entries, field_prefix):
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError(f"{field_prefix} 維度資料數量異常：需為 6")
    by_key = {}
    for item in entries:
        if not isinstance(item, dict):
            raise TypeError(f"{field_prefix} 維度項目必須是物件")
        key = str(item.get("key", "")).strip()
        if key not in DIMENSION_KEYS:
            raise ValueError(f"{field_prefix} 出現未知維度 key：{key}")
        by_key[key] = item
    missing = [k for k in DIMENSION_KEYS if k not in by_key]
    if missing:
        raise ValueError(f"{field_prefix} 缺少維度：{missing}")
    return by_key


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def angle_from_cosine(cos_val):
    cos_val = clamp(cos_val, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


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
    mag_u = abs(user_score)
    mag_b = abs(background_score)
    if abs(mag_u - mag_b) <= 10:
        return "同向近似"
    if mag_u > mag_b:
        return "同向，但本體更強"
    return "同向，但背景更強"


def compact_title(title, max_len=72):
    title = normalize_whitespace(title)
    if len(title) <= max_len:
        return title
    return title[: max_len - 1] + "…"


def escape_latex(text):
    if text is None:
        return ""
    replacements = [
        ("\\", "__LATEX_BACKSLASH__"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ("__LATEX_BACKSLASH__", r"\textbackslash{}"),
    ]
    out = str(text)
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def markdown_to_latex(text):
    lines = str(text).splitlines()
    out = []
    for line in lines:
        if line.startswith("### "):
            out.append(f"\\subsubsection{{{escape_latex(line[4:])}}}")
        elif line.startswith("## "):
            out.append(f"\\subsection{{{escape_latex(line[3:])}}}")
        elif line.startswith("# "):
            out.append(f"\\section{{{escape_latex(line[2:])}}}")
        else:
            out.append(escape_latex(line))
    return "\n".join(out)


def build_dimensions_prompt(manifest):
    payload = []
    for d in DIMENSIONS:
        key = d["key"]
        dim_def = manifest["dimensions"][key]
        payload.append({
            "key": key,
            "zh_label": d["zh"],
            "en_label": d["en"],
            "sin_def": dim_def["sin_def"],
            "cos_def": dim_def["cos_def"],
        })
    return json.dumps(payload, ensure_ascii=False)


def evaluate_user_profile(raw_text, manifest):
    client = get_llm_client()
    manifest_str = build_dimensions_prompt(manifest)

    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論量化儀器」。
請根據文本內容，對 6 個維度做定量評估。

維度定義：
{manifest_str}

量化規則：
1. 每一維都要回傳 signed_score，範圍必須是 -100 到 +100 的整數。
2. +100 = 非常強烈的離群突破（sin）；0 = 中性；-100 = 非常強烈的合群守成（cos）。
3. confidence 範圍必須是 0 到 100 的整數。
4. reason 必須是簡潔中文短語，客觀，不煽情。
5. academic_fingerprint 必須是 60-110 字中文，客觀。
6. core_statement 作為核心宣告與檢索句。必須控制在 10-15 個英文單字，拒絕八股與口號，直指拓樸與演化本質。

請只回傳 JSON：
{{
  "core_statement": "<10-15 字精準英文核心宣告>",
  "academic_fingerprint": "<中文學術指紋>",
  "dimensions": [
    {{"key": "value_intent", "signed_score": 85, "confidence": 92, "reason": "..."}} ,
    {{"key": "governance", "signed_score": 72, "confidence": 88, "reason": "..."}} ,
    {{"key": "cognition", "signed_score": 90, "confidence": 90, "reason": "..."}} ,
    {{"key": "architecture", "signed_score": 83, "confidence": 87, "reason": "..."}} ,
    {{"key": "expansion", "signed_score": 78, "confidence": 85, "reason": "..."}} ,
    {{"key": "application", "signed_score": 68, "confidence": 80, "reason": "..."}}
  ]
}}
""".strip()

    print("🕸️ [階段 1] 量化本體強度向量，並提取 10-15 字精準核心宣告...")
    response = call_llm_with_retry(
        client,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": raw_text[:8000]},
        ],
        temperature=0.0,
        json_mode=True,
    )
    res = parse_llm_json(response.choices[0].message.content)

    core_statement = normalize_whitespace(res.get("core_statement", ""))
    academic_fingerprint = normalize_whitespace(res.get("academic_fingerprint", ""))

    if not core_statement:
        raise ValueError("core_statement 為空。")

    by_key = validate_dimension_entries(res.get("dimensions", []), "本體量化")
    scores = {}
    confidences = {}
    reasons = {}
    for key in DIMENSION_KEYS:
        item = by_key[key]
        scores[key] = enforce_score(item.get("signed_score"), f"{key}.signed_score")
        confidences[key] = enforce_confidence(item.get("confidence"), f"{key}.confidence")
        reasons[key] = normalize_whitespace(item.get("reason", ""))

    return {
        "core_statement": core_statement,
        "academic_fingerprint": academic_fingerprint,
        "scores": scores,
        "confidences": confidences,
        "reasons": reasons,
        "hex_code": sign_to_binary(scores)
    }


def fetch_broad_neighborhood_crossref(core_statement):
    headers = {
        "User-Agent": "AVH-Hologram-Engine/35.4 (https://github.com/alaric-kuo; mailto:open-source-bot@example.com)"
    }
    encoded_query = urllib.parse.quote(core_statement)
    url = f"https://api.crossref.org/works?query={encoded_query}&select=DOI,title,abstract&rows=30"

    print(f"🌍 [階段 2] 投放核心宣告：『{core_statement}』\n🌍 正在 Crossref 禮貌池中打撈關聯文獻...")
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code == 429:
        print("⚠️ Crossref 暫時限流，等待 5 秒後重試...")
        time.sleep(5)
        response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    data = response.json()
    items = data.get("message", {}).get("items", [])

    raw_papers = []
    for paper in items:
        raw_abstract = paper.get("abstract")
        if not raw_abstract:
            continue
        clean_abstract = clean_crossref_abstract(raw_abstract)
        if not clean_abstract:
            continue
        title_list = paper.get("title") or []
        title = title_list[0] if title_list and isinstance(title_list[0], str) else "Unknown"
        doi = str(paper.get("DOI", "Unknown")).strip()

        raw_papers.append({
            "id": doi,
            "title": normalize_whitespace(title),
            "abstract": clean_abstract[:900],
        })
        if len(raw_papers) >= 20:
            break

    print(f"🌍 成功撈取 {len(raw_papers)} 篇具備摘要之文獻，準備進行大腦重排...")
    return raw_papers


def rerank_and_filter_papers(core_statement, raw_papers):
    if not raw_papers:
        return [], "無可用文獻進行重排。"

    client = get_llm_client()
    papers_json = json.dumps(raw_papers, ensure_ascii=False)

    sys_prompt = f"""
你現在是一位客觀的學術觀測員。
本理論唯一核心宣告為："{core_statement}"

請閱讀以下初步打撈的文獻，進行理論結構重排：
1. 嚴格保留真正與這個核心宣告「同題」或「可對話」的文獻。
2. 強制剔除所有只是「撞關鍵字」但領域或問題設定完全無關的文獻（例如探討旅遊、毫無關聯的應用等）。
3. 寧缺勿濫，最多保留 8 篇，如果沒有任何一篇具有理論對話價值，請直接回傳空陣列 []。

請只回傳 JSON：
{{
  "selected_ids": ["<保留的 id>"],
  "filtering_log": "<中文簡述保留與剔除理由>"
}}
""".strip()

    print("⚖️ [階段 3] 啟動結構重排，強制剔除撞字雜訊，萃取純淨對話母體...")
    response = call_llm_with_retry(
        client,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": papers_json},
        ],
        temperature=0.0,
        json_mode=True,
    )
    res = parse_llm_json(response.choices[0].message.content)

    raw_selected = res.get("selected_ids", [])
    if not isinstance(raw_selected, list):
        raise TypeError("selected_ids 必須是陣列")

    valid_ids = {p["id"] for p in raw_papers}
    selected_ids = {str(sid).strip() for sid in raw_selected if str(sid).strip() in valid_ids}
    filtering_log = normalize_whitespace(res.get("filtering_log", "執行標準過濾機制。"))

    final_papers = [p for p in raw_papers if p["id"] in selected_ids][:8]
    return final_papers, filtering_log


def evaluate_background_papers(final_papers, manifest, core_statement):
    if not final_papers:
        return {"papers": [], "batch_log": "無背景文獻可量化。"}

    client = get_llm_client()
    manifest_str = build_dimensions_prompt(manifest)
    papers_str = json.dumps(final_papers, ensure_ascii=False)

    sys_prompt = f"""
你是一台「背景文獻向量量化儀」。
觀測原點唯一核心宣告為："{core_statement}"

請逐篇閱讀以下純淨的對話母體文獻摘要，並用相同的六維座標進行量化。

維度定義：
{manifest_str}

量化規則：
1. 每篇文獻每一維都必須回傳 signed_score (-100 到 +100)。
2. +100 = 非常強烈離群突破（sin）；0 = 中性；-100 = 非常強烈合群守成（cos）。
3. note 用 10-30 字中文簡述該文獻與核心宣告的對位特徵。

請只回傳 JSON：
{{
  "batch_log": "<中文簡述整批背景文獻特徵，60-140字>",
  "papers": [
    {{
      "id": "<doi>",
      "note": "<短中文說明>",
      "scores": [
        {{"key": "value_intent", "signed_score": 25}},
        {{"key": "governance", "signed_score": 10}},
        {{"key": "cognition", "signed_score": 40}},
        {{"key": "architecture", "signed_score": 35}},
        {{"key": "expansion", "signed_score": 20}},
        {{"key": "application", "signed_score": -15}}
      ]
    }}
  ]
}}
""".strip()

    print("📚 [階段 4] 逐篇量化純淨背景文獻強度向量...")
    response = call_llm_with_retry(
        client,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": papers_str},
        ],
        temperature=0.0,
        json_mode=True,
    )
    res = parse_llm_json(response.choices[0].message.content)

    returned = res.get("papers", [])
    valid_map = {p["id"]: p for p in final_papers}
    scored_papers = []

    for item in returned:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id", "")).strip()
        if paper_id not in valid_map:
            continue

        by_key = validate_dimension_entries(item.get("scores", []), f"背景文獻 {paper_id}")
        scores = {}
        for key in DIMENSION_KEYS:
            scores[key] = enforce_score(by_key[key].get("signed_score"), f"{paper_id}.{key}.signed_score")

        scored_papers.append({
            "id": paper_id,
            "title": valid_map[paper_id]["title"],
            "note": normalize_whitespace(item.get("note", "")),
            "scores": scores,
        })

    batch_log = normalize_whitespace(res.get("batch_log", "背景文獻已完成逐篇量化。"))

    if not scored_papers:
        return {
            "papers": [],
            "batch_log": "背景量化回傳空集合，系統無法建立可測量母體。"
        }

    return {"papers": scored_papers, "batch_log": batch_log}


def aggregate_background(scored_papers):
    mean_scores = {}
    peak_scores = {}
    peak_papers = {}
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

    cos_val = cosine_similarity(user_vec, bg_vec)
    angle = round(angle_from_cosine(cos_val), 1)
    global_proximity = round(max(0.0, 100.0 - angle / 1.8), 1)

    if angle < 30:
        global_relation = "高度同向"
    elif angle < 60:
        global_relation = "中度同向"
    elif angle < 90:
        global_relation = "弱同向"
    elif angle == 90:
        global_relation = "正交"
    elif angle < 120:
        global_relation = "弱反向"
    else:
        global_relation = "明顯反向"

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
        peak_compare = "背景峰值更強" if abs(peak) > abs(u) else "本體仍更強"

        vector_logs.append({
            "key": key,
            "label": dim_label(key),
            "user_score": u,
            "background_mean": b,
            "background_peak": peak,
            "peak_title": compact_title(peak_paper["title"]),
            "relation": relation,
            "proximity": proximity,
            "diff_mean": diff_mean,
            "diff_peak": diff_peak,
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
        label = dim_label(key)
        score = user_profile["scores"][key]
        conf = user_profile["confidences"][key]
        reason = user_profile["reasons"][key]
        side = signed_score_to_side(score)
        logs.append(f"* **{label}**：`{score:+d}` / 100 ｜ **{side}** ｜ 置信度 `{conf}` ｜ 觀測判定：{reason}")
    return logs


def format_vector_logs(vector_data):
    logs = []
    for item in vector_data["vector_logs"]:
        logs.append(
            f"* **{item['label']}**：本體 `{item['user_score']:+d}` ｜ 背景均值 `{item['background_mean']:+.1f}` ｜ "
            f"背景峰值 `{item['background_peak']:+d}`（{item['peak_title']}） ｜ "
            f"方向 `{item['relation']}` ｜ 相近度 `{item['proximity']}` / 100 ｜ "
            f"均值差 `{item['diff_mean']:+.1f}` ｜ 峰值差 `{item['diff_peak']:+.1f}` ｜ {item['peak_compare']}"
        )
    return logs


def format_reference_records(scored_papers):
    rows = []
    for p in scored_papers:
        doi_link = f"https://doi.org/{p['id']}" if p["id"] != "Unknown" else "#"
        note = f"｜{p['note']}" if p["note"] else ""
        rows.append(f"- [DOI 連結]({doi_link}) **{p['title']}** {note}")
    return rows


def generate_summary(raw_text, global_relation, global_angle, global_proximity):
    client = get_llm_client()
    prompt = f"""
本理論在外部背景場中的整體關係為：{global_relation}。
整體相位角：約 {global_angle} 度。
整體語意相近度：約 {global_proximity} / 100。

請根據下文，撰寫 180-240 字中文理論導讀。第一句必須以「本理論架構...」開頭。客觀不神話化。
""".strip()

    try:
        response = call_llm_with_retry(
            client,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": raw_text[:5000]},
            ],
            temperature=0.2,
            json_mode=False,
        )
        return zhconv.convert((response.choices[0].message.content or "").strip(), "zh-tw")
    except Exception:
        return f"本理論架構目前與背景場的整體關係為{global_relation}，相位角約為 {global_angle} 度，語意相近度約為 {global_proximity} / 100。由於摘要生成階段遭遇雲端限流，系統暫以保底敘述輸出。"


def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        if len(raw_text.strip()) < 100:
            print("⚠️ 文本過短，略過掃描。")
            return None

        user_profile = evaluate_user_profile(raw_text, manifest)
        user_hex = user_profile["hex_code"]
        state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})

        try:
            raw_papers = fetch_broad_neighborhood_crossref(user_profile["core_statement"])
        except Exception as e:
            raw_papers = []
            filtering_log = f"Crossref 打撈失敗，系統直接回歸無人區（{e}）"
            final_papers = []
        else:
            try:
                final_papers, filtering_log = rerank_and_filter_papers(user_profile["core_statement"], raw_papers)
            except Exception as e:
                final_papers = []
                filtering_log = f"大腦重排失敗，系統直接回歸無人區（{e}）"

        if not final_papers:
            baseline_status = "Void（無人區：外部場域尚不足以形成可測量母體）"
            background_hex = "000000"
            paper_records = ["- `[Void]` **全域寂靜**：周遭尚無足夠背景能勢質量，無法形成可測量母體。"]
            vector_logs = ["* **背景向量量化**：無人區狀態，暫無穩定背景向量可供干涉比較。"]
            global_angle = "無定義（Void）"
            global_cosine = "N/A"
            global_proximity = "N/A"
            global_relation = "無人區"
            background_batch_log = "最終保留文獻為 0，系統判定當前外部場域不足以構成可測量背景母體。"
            summary = "本理論架構目前處於無人區狀態；外部鄰近文獻尚不足以形成穩定背景母體，因此與現有學界的方向關係暫時不可定義。"
        else:
            try:
                scored_background = evaluate_background_papers(final_papers, manifest, user_profile["core_statement"])
            except Exception as e:
                scored_background = {"papers": [], "batch_log": f"背景量化失敗（{e}）"}

            if not scored_background["papers"]:
                baseline_status = "Void（無人區：背景量化回傳空集合，無法形成可測量母體）"
                background_hex = "000000"
                paper_records = ["- `[Void]` **全域寂靜**：背景量化未回傳有效文獻，系統直接回歸無人區。"]
                vector_logs = ["* **背景向量量化**：背景量化結果為空，暫無穩定背景向量可供干涉比較。"]
                global_angle = "無定義（Void）"
                global_cosine = "N/A"
                global_proximity = "N/A"
                global_relation = "無人區"
                background_batch_log = scored_background["batch_log"]
                summary = "本理論架構目前處於無人區狀態；雖然前段曾保留背景文獻，但背景量化未形成有效向量母體，因此方向關係暫時不可定義。"
            else:
                baseline_status = f"Background Field Established（背景能勢建構完成：{len(final_papers)} 鄰域節點）"
                background_batch_log = scored_background["batch_log"]

                vector_data = build_vector_logs(user_profile, scored_background["papers"])
                background_hex = vector_data["background_hex"]
                vector_logs = format_vector_logs(vector_data)
                global_angle = f"{vector_data['global_angle']} 度（{vector_data['global_relation']}）"
                global_cosine = vector_data["global_cosine"]
                global_proximity = vector_data["global_proximity"]
                global_relation = vector_data["global_relation"]
                paper_records = format_referenceRecords(scored_background["papers"]) if False else format_reference_records(scored_background["papers"])

                summary = generate_summary(
                    raw_text, vector_data["global_relation"], vector_data["global_angle"], vector_data["global_proximity"]
                )

        return {
            "user_hex": user_hex,
            "baseline_hex": background_hex,
            "state_name": state_info["name"],
            "state_desc": state_info["desc"],
            "summary": summary,
            "full_text": raw_text,
            "meta_data": {
                "core_statement": user_profile["core_statement"],
                "academic_fingerprint": user_profile["academic_fingerprint"],
                "user_dimension_logs": format_user_dimension_logs(user_profile),
                "raw_hits": len(raw_papers) if 'raw_papers' in locals() else 0,
                "final_hits": len(final_papers) if final_papers else 0,
                "filtering_log": filtering_log,
                "background_batch_log": background_batch_log,
                "paper_records": paper_records,
                "vector_logs": vector_logs,
                "baseline_status": baseline_status,
                "global_angle": global_angle,
                "global_cosine": global_cosine,
                "global_proximity": global_proximity,
                "global_relation": global_relation,
                "llm_model": LLM_MODEL_NAME,
            },
        }
    except Exception as e:
        print(f"❌ 檔案 {source_path} 處理失敗: {e}")
        return None


def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    meta = data["meta_data"]
    user_logs_text = "\n\n".join(meta["user_dimension_logs"])
    papers_text = "\n".join(meta["paper_records"])
    vector_logs_text = "\n\n".join(meta["vector_logs"])

    return (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳（CST）**：`{timestamp}`\n"
        f"* **高維算力引擎**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 0. 🩺 GitHub Models 診斷（Diagnostics）\n"
        f"* **總狀態**：`{DIAG_STATE['status']}`\n"
        f"* **訊息**：`{DIAG_STATE['message']}`\n"
        f"* **純文字 ping**：`{DIAG_STATE['text_ping']}`\n"
        f"* **JSON ping**：`{DIAG_STATE['json_ping']}`\n\n"
        f"---\n"
        f"### 1. 🌌 絕對本體觀測（Absolute Ontology）\n"
        f"* 🛡️ **本體論絕對指紋（Ontology Hex）**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* **本體核心宣告（Core Statement）**：`{meta['core_statement']}`\n\n"
        f"**學術指紋（Academic Fingerprint）**：\n"
        f"> {meta['academic_fingerprint']}\n\n"
        f"**詳細本體量化儀表板（Ontology Quantification Dashboard）**：\n\n"
        f"{user_logs_text}\n\n"
        f"---\n"
        f"### 2. 🎣 背景能勢打撈（Background Field Retrieval）\n"
        f"* **場域建構狀態（Field Status）**：`{meta['baseline_status']}` （原始打撈 {meta['raw_hits']} 篇 → 最終保留 {meta['final_hits']} 篇）\n"
        f"* **大腦重排日誌（Re-ranking Log）**：_{meta['filtering_log']}_\n"
        f"* **背景批次量化摘要（Batch Quantification Log）**：_{meta['background_batch_log']}_\n"
        f"* **參考鄰域節點（Reference Neighborhood）**：\n"
        f"{papers_text}\n\n"
        f"---\n"
        f"### 3. 📐 向量干涉量化（Quantified Vector Interference）\n"
        f"* **背景絕對指紋（Background Hex）**：`[{data['baseline_hex']}]`\n"
        f"* **整體場域關係（Global Relation）**：**{meta['global_relation']}**\n"
        f"* **整體相位角（Global Angle）**：`{meta['global_angle']}`\n"
        f"* **全域餘弦相似（Global Cosine Similarity）**：`{meta['global_cosine']}`\n"
        f"* **整體語意相近度（Global Semantic Proximity）**：`{meta['global_proximity']}` / 100\n"
        f"* **量化公式（Quantification Rule）**：`Per-dimension proximity = 100 - |U - B| / 2; Global angle = arccos(dot(U,B)/(||U||·||B||))`\n\n"
        f"**維度向量干涉儀表板（Per-Dimension Vector Dashboard）**：\n\n"
        f"{vector_logs_text}\n\n"
        f"---\n"
        f"### 4. 🧾 系統導讀摘要（System Interpretation）\n"
        f"> {data['summary']}\n\n"
        f"---\n"
        f"> *註：本報告採 V35.4 診斷穩定版。先驗明 GitHub Models 死因，再執行 AVH 主流程。*\n"
    )


def export_wordpress_html(basename, data):
    safe_full_text = html.escape(data["full_text"]).replace("\n", "<br>")
    safe_summary = html.escape(data["summary"])
    meta = data["meta_data"]
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "  <div class=\"avh-content\">\n"
        f"    {safe_full_text}\n"
        "  </div>\n"
        "  <hr>\n"
        "  <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "    <h3>📡 學術價值全像儀（AVH）診斷穩定認證</h3>\n"
        f"    <p><strong>核心宣告：</strong>{html.escape(meta['core_statement'])}</p>\n"
        f"    <p><strong>本體狀態：</strong>[ {html.escape(data['user_hex'])} ] - {html.escape(data['state_name'])}</p>\n"
        f"    <p><strong>背景狀態：</strong>[ {html.escape(data['baseline_hex'])} ]</p>\n"
        f"    <p><strong>整體場域關係：</strong>{html.escape(str(meta['global_relation']))}</p>\n"
        f"    <p><strong>整體相位角：</strong>{html.escape(str(meta['global_angle']))}</p>\n"
        f"    <p><strong>整體語意相近度：</strong>{html.escape(str(meta['global_proximity']))} / 100</p>\n"
        f"    <p><strong>理論導讀摘要：</strong><br>{safe_summary}</p>\n"
        f"    <p>物理時間戳：{timestamp_str}</p>\n"
        "  </div>\n"
        "</div>\n"
    )
    with open(f"WP_Ready_{basename}.html", "w", encoding="utf-8") as f:
        f.write(html_output)


def export_latex(basename, data):
    safe_text = markdown_to_latex(data["full_text"])
    meta = data["meta_data"]
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        f"\\title{{{escape_latex(basename)}}}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        f"核心宣告：{escape_latex(meta['core_statement'])}\n\n"
        f"本體狀態：[{data['user_hex']}] {escape_latex(data['state_name'])}\n\n"
        f"背景狀態：[{data['baseline_hex']}]\n\n"
        f"整體場域關係：{escape_latex(str(meta['global_relation']))}\n\n"
        f"整體相位角：{escape_latex(str(meta['global_angle']))}\n\n"
        f"整體語意相近度：{escape_latex(str(meta['global_proximity']))}/100\n"
        "\\end{abstract}\n\n"
        f"{safe_text}\n\n"
        "\\end{document}\n"
    )
    with open(f"{basename}_Archive.tex", "w", encoding="utf-8") as f:
        f.write(tex_output)


if __name__ == "__main__":
    if not os.path.exists("avh_manifest.json"):
        print("⚠️ 遺失底層定義檔 avh_manifest.json，終止執行。")
        sys.exit(1)

    with open("avh_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files:
        print("ℹ️ 未找到任何 Markdown 來源檔。")
        sys.exit(0)

    maybe_startup_jitter()
    diag = run_github_model_diagnostics()

    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V35.4 診斷穩定日誌\n---\n")

        if diag["status"] != "OK":
            log_file.write(
                "## 🩺 GitHub Models 診斷結果\n"
                f"* **總狀態**：`{diag['status']}`\n"
                f"* **訊息**：`{diag['message']}`\n"
                f"* **純文字 ping**：`{diag['text_ping']}`\n"
                f"* **JSON ping**：`{diag['json_ping']}`\n\n"
                "---\n"
                "> ❌ 診斷未通過，系統停止後續 AVH 主流程。\n"
            )
            print(f"❌ GitHub Models 診斷失敗：{diag['message']}")
            sys.exit(1)

        last_hex_code = ""
        success_count = 0

        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                success_count += 1
                last_hex_code = result_data["user_hex"]
                log_file.write(generate_trajectory_log(target_source, result_data))
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data)
            else:
                log_file.write(f"\n> ⚠️ `[{target_source}]` 掃描失敗或略過，詳見系統執行日誌。\n---\n")

    if success_count == 0:
        print("❌ 本輪無任何檔案成功完成 AVH 主流程。")
        sys.exit(1)

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a", encoding="utf-8") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
