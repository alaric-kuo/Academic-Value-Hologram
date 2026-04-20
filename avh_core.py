import os
import sys
import json
import glob
import re
import time
import math
import html
import requests
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V34.1 單核精準檢索版 - Core Statement = Query)
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

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "through",
    "by", "with", "from", "into", "via", "using", "use", "based", "beyond",
    "toward", "towards", "within", "across", "under", "over", "between"
}

print(f"🧠 [載入觀測核心] 啟動 V34.1 單核精準檢索版 ({LLM_MODEL_NAME})...")


# ------------------------------------------------------------------------------
# 基礎工具
# ------------------------------------------------------------------------------

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("遺失 GITHUB_TOKEN，無法啟動算力。")
    return OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token,
    )


def call_llm_with_retry(client, messages, temperature=0.0, max_retries=4, json_mode=True):
    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "messages": messages,
                "model": LLM_MODEL_NAME,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"⚠️ 雲端連線異常 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試... [{e}]")
            if attempt < max_retries - 1:
                time.sleep(wait_time)

    raise ConnectionError(f"雲端算力請求超時或阻擋 ({last_error})")


def parse_llm_json(response_text):
    if response_text is None:
        raise ValueError("LLM 未回傳任何內容。")

    text = response_text.strip()
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
        raise ValueError(f"{field_prefix} 維度資料數量異常：需為 6，取得 {len(entries) if isinstance(entries, list) else '非陣列'}")

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
        ("\\", "__LATEX_BACKSLASH__"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
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


def tokenize_query_terms(text):
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", str(text).lower()) if t not in STOPWORDS]


def build_phrase_windows(core_statement):
    tokens = tokenize_query_terms(core_statement)
    windows = []

    # 3-gram 與 2-gram 作為強錨點
    for n in [3, 2]:
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            if phrase not in windows:
                windows.append(phrase)

    return windows[:8]


# ------------------------------------------------------------------------------
# 階段 1：本體量化 + 單核檢索句生成
# ------------------------------------------------------------------------------

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
2. +100 = 非常強烈的離群突破（sin）。
3. 0 = 中性、混合或不足以判定。
4. -100 = 非常強烈的合群守成（cos）。
5. confidence 範圍必須是 0 到 100 的整數。
6. reason 必須是簡潔中文短語，客觀，不煽情。
7. academic_fingerprint 必須是 60-110 字中文，客觀，不要鼓勵語氣，不要神話化。

最重要：
8. core_statement 同時是「展示主題句」與「檢索句」，不可拆成兩套。
9. 它必須是可檢索、可追溯、精準的英文學術句，不是口號，不是標題，不是漂亮摘要。
10. 長度以 8-18 個英文詞為原則。
11. 必須盡量包含：研究對象、方法框架、判準衝突 或 目標問題。
12. 避免過度泛化詞，例如只寫 redefining / evolution / trust / ontology 而沒有具體對象。

請只回傳 JSON：
{{
  "core_statement": "<同時作為展示與檢索的英文學術句>",
  "academic_fingerprint": "<60-110字中文學術指紋>",
  "dimensions": [
    {{
      "key": "value_intent",
      "signed_score": 85,
      "confidence": 92,
      "reason": "重構知識邊界"
    }},
    {{
      "key": "governance",
      "signed_score": 72,
      "confidence": 88,
      "reason": "突破剛性治理框架"
    }},
    {{
      "key": "cognition",
      "signed_score": 90,
      "confidence": 90,
      "reason": "直指本體與混亂核心"
    }},
    {{
      "key": "architecture",
      "signed_score": 83,
      "confidence": 87,
      "reason": "全域連續動態架構"
    }},
    {{
      "key": "expansion",
      "signed_score": 78,
      "confidence": 85,
      "reason": "提取跨域通用協議"
    }},
    {{
      "key": "application",
      "signed_score": 68,
      "confidence": 80,
      "reason": "轉化為現實干預工具"
    }}
  ]
}}
""".strip()

    print("🕸️ [階段 1] 量化本體強度向量，並生成單核精準檢索句...")
    response = call_llm_with_retry(
        client,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": raw_text[:9000]},
        ],
        temperature=0.0,
        json_mode=True,
    )
    res = parse_llm_json(response.choices[0].message.content)

    core_statement = normalize_whitespace(res.get("core_statement", ""))
    academic_fingerprint = normalize_whitespace(res.get("academic_fingerprint", ""))

    if not core_statement:
        raise ValueError("core_statement 為空。")
    if not academic_fingerprint:
        raise ValueError("academic_fingerprint 為空。")

    by_key = validate_dimension_entries(res.get("dimensions", []), "本體量化")

    scores = {}
    confidences = {}
    reasons = {}
    for key in DIMENSION_KEYS:
        item = by_key[key]
        scores[key] = enforce_score(item.get("signed_score"), f"{key}.signed_score")
        confidences[key] = enforce_confidence(item.get("confidence"), f"{key}.confidence")
        reasons[key] = normalize_whitespace(item.get("reason", ""))

    query_terms = list(dict.fromkeys(tokenize_query_terms(core_statement)))
    query_phrases = build_phrase_windows(core_statement)

    return {
        "core_statement": core_statement,
        "academic_fingerprint": academic_fingerprint,
        "scores": scores,
        "confidences": confidences,
        "reasons": reasons,
        "hex_code": sign_to_binary(scores),
        "query_terms": query_terms,
        "query_phrases": query_phrases,
    }


# ------------------------------------------------------------------------------
# 階段 2：Crossref 打撈
# ------------------------------------------------------------------------------

def fetch_broad_neighborhood_crossref(core_statement):
    headers = {
        "User-Agent": "AVH-Hologram-Engine/34.1 (https://github.com/alaric-kuo; mailto:open-source-bot@example.com)"
    }
    params = {
        "query": core_statement,
        "select": "DOI,title,abstract",
        "rows": 30,
    }

    print(f"🌍 [階段 2] 投放核心宣告（同時作為檢索句）：『{core_statement}』")
    print("🌍 正在 Crossref 禮貌池中打撈關聯文獻...")

    try:
        response = requests.get(
            "https://api.crossref.org/works",
            headers=headers,
            params=params,
            timeout=20,
        )

        if response.status_code == 429:
            print("⚠️ 遭遇 Crossref 瞬間限流，強制退避 5 秒...")
            time.sleep(5)
            response = requests.get(
                "https://api.crossref.org/works",
                headers=headers,
                params=params,
                timeout=20,
            )

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

            if len(raw_papers) >= 24:
                break

        print(f"🌍 成功撈取 {len(raw_papers)} 篇具備摘要之文獻，準備進行 lexical prefilter...")
        return raw_papers

    except Exception as e:
        raise ConnectionError(f"Crossref 連線異常或超時 ({e})")


# ------------------------------------------------------------------------------
# 階段 3：程式級 lexical prefilter
# ------------------------------------------------------------------------------

def score_paper_by_core_statement(paper, core_statement, query_terms, query_phrases):
    combined = normalize_whitespace(f"{paper['title']} {paper['abstract']}").lower()

    score = 0
    matched_terms = []
    matched_phrases = []

    full_statement = normalize_whitespace(core_statement).lower()
    if full_statement and full_statement in combined:
        score += 45
        matched_phrases.append(full_statement)

    for phrase in query_phrases:
        if phrase and phrase in combined:
            score += 18
            matched_phrases.append(phrase)

    for term in query_terms:
        if term in combined:
            score += 8
            matched_terms.append(term)

    coverage = 0.0
    if query_terms:
        coverage = len(set(matched_terms)) / len(set(query_terms))

    # 標題命中加權
    title_l = paper["title"].lower()
    title_hits = sum(1 for term in query_terms if term in title_l)
    score += title_hits * 5

    return {
        "lexical_score": score,
        "coverage": round(coverage, 3),
        "matched_terms": sorted(set(matched_terms)),
        "matched_phrases": sorted(set(matched_phrases)),
    }


def prefilter_raw_papers(raw_papers, core_statement, query_terms, query_phrases, keep_top_k=12):
    if not raw_papers:
        return [], "無原始文獻可做 lexical prefilter。"

    enriched = []
    for paper in raw_papers:
        scored = score_paper_by_core_statement(paper, core_statement, query_terms, query_phrases)
        p2 = dict(paper)
        p2.update(scored)
        enriched.append(p2)

    enriched.sort(
        key=lambda x: (x["lexical_score"], x["coverage"], len(x["matched_terms"])),
        reverse=True
    )

    kept = [p for p in enriched if p["lexical_score"] > 0][:keep_top_k]

    if not kept and enriched:
        kept = enriched[:min(6, len(enriched))]

    if kept:
        top = kept[0]
        log = (
            f"程式級 prefilter 已啟動；原始 {len(raw_papers)} 篇，保留 {len(kept)} 篇。"
            f"最高 lexical score = {top['lexical_score']}，coverage = {top['coverage']:.3f}。"
            f"主要命中詞：{', '.join(top['matched_terms'][:8]) if top['matched_terms'] else '無'}。"
        )
    else:
        log = f"程式級 prefilter 已啟動；原始 {len(raw_papers)} 篇，但無任何文獻命中核心檢索句。"

    return kept, log


# ------------------------------------------------------------------------------
# 階段 4：LLM 重排
# ------------------------------------------------------------------------------

def rerank_and_filter_papers(core_statement, prefiltered_papers):
    if not prefiltered_papers:
        return [], "無可用文獻進行重排。"

    client = get_llm_client()
    papers_json = json.dumps(prefiltered_papers, ensure_ascii=False)

    sys_prompt = f"""
你現在是一位客觀的學術觀測員。
本理論唯一核心宣告（同時作為檢索句）為：
"{core_statement}"

以下文獻已經通過程式級 lexical prefilter。
請你再做一次理論結構重排：
1. 保留真正與這個核心宣告同題或可對話的文獻。
2. 剔除只是撞字、但問題設定不同的文獻。
3. 最多保留 8 篇，0 篇則回傳空陣列。
4. filtering_log 必須用中文，明確說明保留與剔除判準。

請只回傳 JSON：
{{
  "selected_ids": ["<保留的 id>"],
  "filtering_log": "<中文簡述保留與剔除理由，80-180字>"
}}
""".strip()

    print("⚖️ [階段 4] 啟動結構重排，萃取真正可對話的背景文獻...")
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
        raise TypeError("selected_ids 必須是陣列 (list)")

    valid_ids = {p["id"] for p in prefiltered_papers}
    selected_ids = {str(sid).strip() for sid in raw_selected if str(sid).strip() in valid_ids}
    filtering_log = normalize_whitespace(res.get("filtering_log", "執行標準過濾機制。"))

    final_papers = [p for p in prefiltered_papers if p["id"] in selected_ids][:8]
    return final_papers, filtering_log


# ------------------------------------------------------------------------------
# 階段 5：背景逐篇量化
# ------------------------------------------------------------------------------

def evaluate_background_papers(final_papers, manifest, core_statement):
    if not final_papers:
        return {"papers": [], "batch_log": "無背景文獻可量化。"}

    client = get_llm_client()
    manifest_str = build_dimensions_prompt(manifest)
    papers_str = json.dumps(final_papers, ensure_ascii=False)

    sys_prompt = f"""
你是一台「背景文獻向量量化儀」。
觀測原點唯一核心宣告為：
"{core_statement}"

請逐篇閱讀以下文獻摘要，並用與本體相同的六維座標進行量化。

維度定義：
{manifest_str}

量化規則：
1. 每篇文獻、每一維都必須回傳 signed_score，範圍是 -100 到 +100 的整數。
2. +100 = 非常強烈離群突破（sin）；0 = 中性；-100 = 非常強烈合群守成（cos）。
3. note 請用 10-30 字中文簡述該文獻與核心宣告的對位特徵。
4. 不要使用鼓勵語氣，不要神話化。

請只回傳 JSON：
{{
  "batch_log": "<中文簡述整批背景文獻的共同特徵與限制，60-140字>",
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

    print("📚 [階段 5] 逐篇量化背景文獻強度向量...")
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
    if not isinstance(returned, list):
        raise TypeError("背景量化回傳 papers 必須是陣列。")

    valid_map = {p["id"]: p for p in final_papers}
    scored_papers = []

    for item in returned:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id", "")).strip()
        if paper_id not in valid_map:
            continue

        scores_entries = item.get("scores", [])
        by_key = validate_dimension_entries(scores_entries, f"背景文獻 {paper_id}")

        scores = {}
        for key in DIMENSION_KEYS:
            scores[key] = enforce_score(by_key[key].get("signed_score"), f"{paper_id}.{key}.signed_score")

        scored_papers.append({
            "id": paper_id,
            "title": valid_map[paper_id]["title"],
            "abstract": valid_map[paper_id]["abstract"],
            "note": normalize_whitespace(item.get("note", "")),
            "lexical_score": valid_map[paper_id].get("lexical_score", 0),
            "coverage": valid_map[paper_id].get("coverage", 0.0),
            "matched_terms": valid_map[paper_id].get("matched_terms", []),
            "matched_phrases": valid_map[paper_id].get("matched_phrases", []),
            "scores": scores,
        })

    if not scored_papers:
        raise ValueError("背景量化結果為空，無法建立背景向量。")

    batch_log = normalize_whitespace(res.get("batch_log", "背景文獻已完成逐篇量化。"))

    return {
        "papers": scored_papers,
        "batch_log": batch_log,
    }


# ------------------------------------------------------------------------------
# 階段 6：幾何量化與可解釋輸出
# ------------------------------------------------------------------------------

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
        peak_title = compact_title(peak_paper["title"])

        vector_logs.append({
            "key": key,
            "label": dim_label(key),
            "user_score": u,
            "background_mean": b,
            "background_peak": peak,
            "peak_title": peak_title,
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


# ------------------------------------------------------------------------------
# 輸出格式化
# ------------------------------------------------------------------------------

def format_user_dimension_logs(user_profile):
    logs = []
    for key in DIMENSION_KEYS:
        label = dim_label(key)
        score = user_profile["scores"][key]
        conf = user_profile["confidences"][key]
        reason = user_profile["reasons"][key]
        side = signed_score_to_side(score)
        logs.append(
            f"* **{label}**：`{score:+d}` / 100 ｜ **{side}** ｜ 置信度 `{conf}` ｜ 觀測判定：{reason}"
        )
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
        terms = ", ".join(p["matched_terms"][:8]) if p["matched_terms"] else "無"
        phrases = "; ".join(p["matched_phrases"][:4]) if p["matched_phrases"] else "無"
        note = f"｜{p['note']}" if p["note"] else ""

        rows.append(
            f"- [DOI 連結]({doi_link}) **{p['title']}** "
            f"｜ lexical `{p['lexical_score']}` ｜ coverage `{p['coverage']}` "
            f"｜ 命中詞 `{terms}` ｜ 命中片語 `{phrases}` {note}"
        )
    return rows


def generate_summary(raw_text, global_relation, global_angle, global_proximity):
    client = get_llm_client()
    prompt = f"""
本理論在外部背景場中的整體關係為：{global_relation}。
整體相位角：約 {global_angle} 度。
整體語意相近度：約 {global_proximity} / 100。

請根據下文，撰寫 180-240 字中文理論導讀。
要求：
1. 第一句必須以「本理論架構...」開頭。
2. 客觀，不要鼓勵語氣。
3. 要指出它與背景場是同向、弱同向、正交還是反向，並說明它強在哪裡、距離在哪裡。
4. 不要神話化，不要空泛。
""".strip()

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


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------

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

        state_info = manifest["states"].get(
            user_hex,
            {"name": "未知狀態", "desc": "缺乏觀測紀錄"}
        )
        user_state_name = state_info["name"]
        user_state_desc = state_info["desc"]

        raw_papers = fetch_broad_neighborhood_crossref(user_profile["core_statement"])
        prefiltered_papers, prefilter_log = prefilter_raw_papers(
            raw_papers,
            user_profile["core_statement"],
            user_profile["query_terms"],
            user_profile["query_phrases"],
            keep_top_k=12
        )
        final_papers, filtering_log = rerank_and_filter_papers(user_profile["core_statement"], prefiltered_papers)

        if not final_papers:
            baseline_status = "Sparse Reference Field（稀疏參考場）"
            background_hex = "000000"
            paper_records = ["- `[Void]` **全域寂靜**：目前不足以構成穩定可量化的背景能勢場。"]
            vector_logs = ["* **背景向量量化**：無足夠背景質量，無法形成穩定比較。"]
            global_angle = "N/A"
            global_cosine = "N/A"
            global_proximity = "N/A"
            global_relation = "Void"
            background_batch_log = "無可用背景文獻。"
            summary = "本理論架構目前落在稀疏參考場之中，外部文獻鄰域不足，尚無法形成穩定背景母體，因此其與現有學界的方向關係暫時未定。"
        else:
            baseline_status = f"Background Field Established（背景能勢建構完成：{len(final_papers)} 鄰域節點）"
            scored_background = evaluate_background_papers(final_papers, manifest, user_profile["core_statement"])
            background_batch_log = scored_background["batch_log"]

            vector_data = build_vector_logs(user_profile, scored_background["papers"])
            background_hex = vector_data["background_hex"]
            vector_logs = format_vector_logs(vector_data)

            global_angle = f"{vector_data['global_angle']} 度（{vector_data['global_relation']}）"
            global_cosine = vector_data["global_cosine"]
            global_proximity = vector_data["global_proximity"]
            global_relation = vector_data["global_relation"]

            paper_records = format_reference_records(scored_background["papers"])

            summary = generate_summary(
                raw_text,
                vector_data["global_relation"],
                vector_data["global_angle"],
                vector_data["global_proximity"],
            )

        return {
            "user_hex": user_hex,
            "baseline_hex": background_hex,
            "state_name": user_state_name,
            "state_desc": user_state_desc,
            "summary": summary,
            "full_text": raw_text,
            "meta_data": {
                "core_statement": user_profile["core_statement"],
                "query_terms": user_profile["query_terms"],
                "query_phrases": user_profile["query_phrases"],
                "academic_fingerprint": user_profile["academic_fingerprint"],
                "user_dimension_logs": format_user_dimension_logs(user_profile),
                "raw_hits": len(raw_papers),
                "prefilter_hits": len(prefiltered_papers),
                "final_hits": len(final_papers),
                "prefilter_log": prefilter_log,
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


# ------------------------------------------------------------------------------
# 產出檔案
# ------------------------------------------------------------------------------

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    meta = data["meta_data"]

    user_logs_text = "\n\n".join(meta["user_dimension_logs"])
    papers_text = "\n".join(meta["paper_records"])
    vector_logs_text = "\n\n".join(meta["vector_logs"])
    query_terms_text = ", ".join(meta["query_terms"]) if meta["query_terms"] else "無"
    query_phrases_text = " ｜ ".join(meta["query_phrases"]) if meta["query_phrases"] else "無"

    return (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳（CST）**：`{timestamp}`\n"
        f"* **高維算力引擎（High-Dimensional Engine）**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 絕對本體觀測（Absolute Ontology）\n"
        f"* 🛡️ **本體論絕對指紋（Ontology Hex）**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* **本體核心宣告／唯一檢索句（Core Statement / Query）**：`{meta['core_statement']}`\n"
        f"* **檢索詞錨點（Query Term Anchors）**：`{query_terms_text}`\n"
        f"* **檢索片語錨點（Query Phrase Anchors）**：`{query_phrases_text}`\n\n"
        f"**學術指紋（Academic Fingerprint）**：\n"
        f"> {meta['academic_fingerprint']}\n\n"
        f"**詳細本體量化儀表板（Ontology Quantification Dashboard）**：\n\n"
        f"{user_logs_text}\n\n"
        f"---\n"
        f"### 2. 🎣 背景能勢打撈（Background Field Retrieval）\n"
        f"* **場域建構狀態（Field Status）**：`{meta['baseline_status']}` （原始打撈 {meta['raw_hits']} 篇 → prefilter 保留 {meta['prefilter_hits']} 篇 → 最終保留 {meta['final_hits']} 篇）\n"
        f"* **程式級預過濾日誌（Programmatic Prefilter Log）**：_{meta['prefilter_log']}_\n"
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
        f"> *註：本報告採 V34.1 單核精準檢索版。Core Statement 與 Query 不分離；檢索 trace 已顯性保留。*\n"
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
        "    <h3>📡 學術價值全像儀（AVH）單核精準檢索認證</h3>\n"
        f"    <p><strong>核心宣告／檢索句：</strong>{html.escape(meta['core_statement'])}</p>\n"
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
        f"核心宣告／檢索句：{escape_latex(meta['core_statement'])}\n\n"
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


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

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

    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V34.1 單核精準檢索日誌\n---\n")
        last_hex_code = ""

        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data["user_hex"]
                log_file.write(generate_trajectory_log(target_source, result_data))
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data)
            else:
                log_file.write(f"\n> ⚠️ `[{target_source}]` 掃描失敗或略過，詳見系統執行日誌。\n---\n")

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a", encoding="utf-8") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
