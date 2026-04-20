import os
import sys
import json
import glob
import time
import math
import requests
import urllib.parse
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V36.4 降維穿透裝甲版 - 標準對話框輸出)
# ==============================================================================

LLM_MODEL_NAME = "openai/gpt-4o"

print(f"🧠 [載入觀測核心] 啟動 V36.4 穩定防爆與限流防禦版 ({LLM_MODEL_NAME})...")

if not os.path.exists("avh_manifest.json"):
    print("工具調用失敗，原因為 遺失底層定義檔 avh_manifest.json")
    sys.exit(1)

with open("avh_manifest.json", "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

DIMENSION_KEYS = list(MANIFEST["dimensions"].keys())

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN，無法啟動算力。")
        sys.exit(1)
    base_u = "ht" + "tps://" + "models.github.ai/inference"
    return OpenAI(base_url=base_u, api_key=token)

def call_llm_with_retry(client, messages, temperature=0.0, max_retries=5, json_mode=True):
    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = {"messages": messages, "model": LLM_MODEL_NAME, "temperature": temperature}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            wait_time = 5 * (2 ** attempt)
            print(f"⚠️ 雲端連線異常或限流 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試... [{e}]")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
    
    print(f"工具調用失敗，原因為 雲端算力請求超時或阻擋 ({last_error})")
    sys.exit(1)

def parse_llm_json(response_text):
    if response_text is None:
        raise ValueError("LLM 未回傳任何內容。")
    
    text = response_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx == -1 or end_idx == -1:
        print("工具調用失敗，原因為 找不到 JSON 邊界符號。")
        sys.exit(1)
        
    clean_json = text[start_idx:end_idx + 1]
    
    try:
        return json.loads(clean_json, strict=False)
    except Exception as e:
        print(f"工具調用失敗，原因為 JSON 解析異常 ({e})")
        sys.exit(1)

def clamp(value, low, high):
    return max(low, min(high, value))

def dim_label(key):
    return MANIFEST["dimensions"][key]["layer"]

def sign_to_binary(scores_by_key):
    return "".join("1" if scores_by_key[k] > 0 else "0" for k in DIMENSION_KEYS)

def signed_score_to_side(score):
    return "離群突破（sin）" if score > 0 else "合群守成（cos）"

def enforce_score(value, field_name):
    try: score = int(round(float(value)))
    except Exception: raise ValueError(f"{field_name} 分數無法解析：{value}")
    return clamp(score, -100, 100)

def enforce_confidence(value, field_name):
    try: conf = int(round(float(value)))
    except Exception: raise ValueError(f"{field_name} 置信度無法解析：{value}")
    return clamp(conf, 0, 100)

def validate_dimension_entries(entries, field_prefix):
    if not isinstance(entries, list) or len(entries) != len(DIMENSION_KEYS):
        print(f"工具調用失敗，原因為 {field_prefix} 維度資料數量異常")
        sys.exit(1)
    by_key = {str(item.get("key", "")).strip(): item for item in entries if isinstance(item, dict)}
    missing = [k for k in DIMENSION_KEYS if k not in by_key]
    if missing: 
        print(f"工具調用失敗，原因為 {field_prefix} 缺少維度：{missing}")
        sys.exit(1)
    return by_key

def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def angle_from_cosine(cos_val):
    return math.degrees(math.acos(clamp(cos_val, -1.0, 1.0)))

def proximity_from_scores(user_score, background_score):
    diff = abs(user_score - background_score)
    return round(max(0.0, 100.0 - diff / 2.0), 1)

def classify_relation(user_score, background_score):
    if abs(background_score) < 10: return "弱耦合"
    if user_score == 0 and background_score == 0: return "中性"
    if user_score * background_score < 0: return "反向干涉"
    mag_u, mag_b = abs(user_score), abs(background_score)
    if abs(mag_u - mag_b) <= 10: return "同向近似"
    return "同向演化 (本體能勢突破)" if mag_u > mag_b else "同向演化 (本體能勢追擊)"

def build_dimensions_prompt():
    payload = [{"key": k, "layer": MANIFEST["dimensions"][k]["layer"], "sin": MANIFEST["dimensions"][k]["sin_def"], "cos": MANIFEST["dimensions"][k]["cos_def"]} for k in DIMENSION_KEYS]
    return json.dumps(payload, ensure_ascii=False)

def evaluate_user_profile(raw_text):
    client = get_llm_client()
    manifest_str = build_dimensions_prompt()
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論量化儀器」。請根據文本內容，對 6 個維度做定量評估。
維度定義：{manifest_str}

【術語強制規範】
1. 絕對禁止使用簡體中文慣用語。
2. 必須使用台灣繁體學術語彙：例如「資訊」（禁「信息」）、「網路」（禁「網絡」）、「巨觀」（禁「宏觀」）。

量化規則：
1. 每一維都要回傳 signed_score (-100 到 +100)。
2. core_statement 控制在 10-15 個英文單字，直指拓樸與演化本質。
3. academic_fingerprint 必須是 60-110 字的動態演化觀測，嚴禁簡體詞彙。

請只回傳 JSON：
{{
  "core_statement": "英文宣告",
  "academic_fingerprint": "繁體中文觀測",
  "dimensions": [
    {{"key": "value_intent", "signed_score": 85, "confidence": 92, "reason": "說明"}}
  ]
}}
"""
    
    print("🕸️ [階段 1] 量化本體強度向量...")
    response = call_llm_with_retry(client, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:8000]}], temperature=0.0)
    res = parse_llm_json(response.choices[0].message.content)
    
    by_key = validate_dimension_entries(res.get("dimensions", []), "本體量化")
    return {
        "core_statement": str(res.get("core_statement", "Academic Ontology")).strip(),
        "academic_fingerprint": str(res.get("academic_fingerprint", "")).strip(),
        "scores": {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS},
        "confidences": {k: enforce_confidence(by_key[k].get("confidence"), k) for k in DIMENSION_KEYS},
        "reasons": {k: str(by_key[k].get("reason", "")).strip() for k in DIMENSION_KEYS},
        "hex_code": sign_to_binary({k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS})
    }

def fetch_broad_neighborhood_crossref(core_statement):
    headers = {"User-Agent": "AVH-Engine/36.4 (mailto:bot@example.com)"}
    encoded_query = urllib.parse.quote(core_statement)
    url = "ht" + "tps://" + f"api.crossref.org/works?query={encoded_query}&select=DOI,title,abstract&rows=30"
    
    print(f"🌍 [階段 2] 投放核心宣告：『{core_statement}』\n🌍 正在 Crossref 穩定池中打撈關聯文獻...")
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 429:
            print("⚠️ Crossref 伺服器限流，退避 5 秒...")
            time.sleep(5)
            response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        
        raw_papers = []
        for paper in items:
            raw_abs = paper.get("abstract", "")
            if not raw_abs: continue
            
            clean_abs = raw_abs.replace("<jats:title>", "").replace("</jats:title>", "")
            clean_abs = clean_abs.replace("<jats:p>", "").replace("</jats:p>", "")
            title = (paper.get("title") or ["Unknown"])[0]
            
            raw_papers.append({
                "id": str(paper.get("DOI", "Unknown")), 
                "title": str(title).strip(), 
                "abstract": clean_abs[:900]
            })
            if len(raw_papers) >= 20: break
        return raw_papers
    except Exception as e: 
        print(f"工具調用失敗，原因為 Crossref 連線異常 ({e})")
        sys.exit(1)

def rerank_and_filter_papers(core_statement, raw_papers):
    if not raw_papers: return [], "無文獻。"
    client, papers_json = get_llm_client(), json.dumps(raw_papers, ensure_ascii=False)
    
    sys_prompt = f"""
你是一位客觀的高維度學術觀測員。唯一核心宣告："{core_statement}"。
請篩選出底層邏輯同構或可深度對話的文獻，強制剔除撞單字但無關者。最多 8 篇。
【術語規範】絕對禁止使用「信息、網絡、宏觀」，必須使用「資訊、網路、巨觀」。

請只回傳 JSON：
{{
  "selected_ids": ["id_1", "id_2"],
  "filtering_log": "繁體中文理由"
}}
"""
    
    print("⚖️ [階段 3] 啟動結構重排...")
    response = call_llm_with_retry(client, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_json}])
    res = parse_llm_json(response.choices[0].message.content)
    
    selected_ids = {str(sid).strip() for sid in res.get("selected_ids", [])}
    return [p for p in raw_papers if p["id"] in selected_ids][:8], str(res.get("filtering_log", "")).strip()

def evaluate_background_papers(final_papers, core_statement):
    if not final_papers: return {"papers": [], "batch_log": "無背景文獻。"}
    client, manifest_str, papers_str = get_llm_client(), build_dimensions_prompt(), json.dumps(final_papers, ensure_ascii=False)
    
    sys_prompt = f"""
你是一台「背景文獻量化儀」。核心宣告："{core_statement}"。
請用相同六維座標量化對話母體。維度定義：{manifest_str}
【術語規範】禁簡體，禁「信息、網絡、宏觀」。
量化規則：每一維回傳 signed_score (-100 到 +100)。note 用 10-30 字繁體中文簡述。

請只回傳 JSON：
{{
  "batch_log": "繁體中文特徵",
  "papers": [
    {{
      "id": "doi", "note": "說明",
      "scores": [ {{"key": "value_intent", "signed_score": 25}} ]
    }}
  ]
}}
"""
    
    print("📚 [階段 4] 逐篇量化背景文獻...")
    response = call_llm_with_retry(client, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}])
    res = parse_llm_json(response.choices[0].message.content)
    
    scored_papers = []
    valid_map = {p["id"]: p for p in final_papers}
    for item in res.get("papers", []):
        p_id = str(item.get("id", "")).strip()
        if p_id not in valid_map: continue
        by_k = validate_dimension_entries(item.get("scores", []), p_id)
        scored_papers.append({
            "id": p_id, "title": valid_map[p_id]["title"], 
            "note": str(item.get("note", "")).strip(), 
            "scores": {k: enforce_score(by_k[k].get("signed_score"), k) for k in DIMENSION_KEYS}
        })
    return {"papers": scored_papers, "batch_log": str(res.get("batch_log", "")).strip()}

def build_vector_logs(user_profile, scored_papers):
    user_scores = user_profile["scores"]
    mean_scores, peak_scores, peak_papers = {}, {}, {}
    for key in DIMENSION_KEYS:
        vals = [(p["scores"][key], p) for p in scored_papers]
        mean_scores[key] = round(sum(v for v, _ in vals) / len(vals), 1)
        peak_val, peak_p = max(vals, key=lambda x: x[0])
        peak_scores[key], peak_papers[key] = peak_val, peak_p
    
    background_hex = sign_to_binary({k: mean_scores[k] for k in DIMENSION_KEYS})
    user_vec, bg_vec = [user_scores[k] for k in DIMENSION_KEYS], [mean_scores[k] for k in DIMENSION_KEYS]
    cos_val = cosine_similarity(user_vec, bg_vec)
    angle = round(angle_from_cosine(cos_val), 1)
    global_proximity = round(max(0.0, 100.0 - angle / 1.8), 1)
    
    if angle < 30: global_relation = "高度同向"
    elif angle < 60: global_relation = "中度同向"
    elif angle < 90: global_relation = "弱同向"
    elif angle == 90: global_relation = "正交"
    elif angle < 120: global_relation = "弱反向"
    else: global_relation = "明顯反向"

    v_logs = []
    for k in DIMENSION_KEYS:
        u, b, pk = user_scores[k], mean_scores[k], peak_scores[k]
        peak_compare = "本體能勢突破" if abs(u) > abs(pk) else "本體能勢追擊"
        v_logs.append({
            "key": k, "label": dim_label(k), "user": u, "mean": b, "peak": pk, "pk_title": peak_papers[k]["title"][:40]+"...",
            "relation": classify_relation(u, b), "proximity": proximity_from_scores(u, b),
            "diff_m": round(u-b, 1), "diff_p": round(u-pk, 1), "compare": peak_compare
        })
    return {"background_hex": background_hex, "global_angle": angle, "global_cosine": round(cos_val, 4), "global_proximity": global_proximity, "global_relation": global_relation, "vector_logs": v_logs}

def format_user_dimension_logs(user_profile):
    return [f"* **{dim_label(k)}**：`{user_profile['scores'][k]:+d}` / 100 ｜ **{signed_score_to_side(user_profile['scores'][k])}** ｜ 置信度 `{user_profile['confidences'][k]}` ｜ 觀測判定：{user_profile['reasons'][k]}" for k in DIMENSION_KEYS]

def format_vector_logs(vector_data):
    return [f"* **{i['label']}**：本體 `{i['user']:+d}` ｜ 背景均值 `{i['mean']:+.1f}` ｜ 背景峰值 `{i['peak']:+d}`（{i['pk_title']}） ｜ 方向 `{i['relation']}` ｜ 相近度 `{i['proximity']}` / 100 ｜ 均值差 `{i['diff_m']:+.1f}` ｜ 峰值差 `{i['diff_p']:+.1f}` ｜ {i['compare']}" for i in vector_data["vector_logs"]]

def format_reference_records(scored_papers):
    return [f"- [DOI] {p['id']} **{p['title']}** ｜{p['note']}" for p in scored_papers]

def generate_summary(raw_text, rel, angle, proximity):
    client = get_llm_client()
    prompt = f"""
整體關係為：{rel}。相位角：約 {angle} 度。語意相近度：約 {proximity} / 100。
請根據下文，撰寫 180-240 字繁體中文理論導讀。第一句必須以「本理論架構...」開頭。客觀不神話化。
【術語規範】絕對禁止使用「信息、網絡、宏觀」，必須使用「資訊、網路、巨觀」。
""".strip()
    response = call_llm_with_retry(client, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": raw_text[:5000]}], temperature=0.2, json_mode=False)
    return zhconv.convert((response.choices[0].message.content or "").strip(), "zh-tw")

def process_avh_manifestation(source_path):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, "r", encoding="utf-8") as file: raw_text = file.read()
        if len(raw_text.strip()) < 100: return None
        user = evaluate_user_profile(raw_text)
        
        raw_papers = fetch_broad_neighborhood_crossref(user["core_statement"])
        final_papers, filtering_log = rerank_and_filter_papers(user["core_statement"], raw_papers)
        
        if not final_papers:
            return {"user_hex": user["hex_code"], "baseline_hex": "000000", "state_name": MANIFEST["states"].get(user["hex_code"], {}).get("name", ""), "state_desc": MANIFEST["states"].get(user["hex_code"], {}).get("desc", ""), "summary": "無人區狀態。", "full_text": raw_text, "meta_data": {"core_statement": user["core_statement"], "academic_fingerprint": user["academic_fingerprint"], "user_dimension_logs": format_user_dimension_logs(user), "raw_hits": len(raw_papers), "final_hits": 0, "filtering_log": filtering_log, "baseline_status": "Void", "global_angle": "無定義", "global_relation": "無人區", "llm_model": LLM_MODEL_NAME}}
        
        scored_bg = evaluate_background_papers(final_papers, user["core_statement"])
        vec = build_vector_logs(user, scored_bg["papers"])
        summary = generate_summary(raw_text, vec["global_relation"], vec["global_angle"], vec["global_proximity"])
        
        return {"user_hex": user["hex_code"], "baseline_hex": vec["background_hex"], "state_name": MANIFEST["states"].get(user["hex_code"], {}).get("name", ""), "state_desc": MANIFEST["states"].get(user["hex_code"], {}).get("desc", ""), "summary": summary, "full_text": raw_text, "meta_data": {"core_statement": user["core_statement"], "academic_fingerprint": user["academic_fingerprint"], "user_dimension_logs": format_user_dimension_logs(user), "raw_hits": len(raw_papers), "final_hits": len(final_papers), "filtering_log": filtering_log, "background_batch_log": scored_bg["batch_log"], "paper_records": format_reference_records(scored_bg["papers"]), "vector_logs": format_vector_logs(vec), "baseline_status": "Established", "global_angle": f"{vec['global_angle']} 度", "global_cosine": vec["global_cosine"], "global_proximity": vec["global_proximity"], "global_relation": vec["global_relation"], "llm_model": LLM_MODEL_NAME}}
    except Exception as e:
        print(f"工具調用失敗，原因為 {source_path} 處理異常: {e}")
        return None

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    meta = data["meta_data"]
    return (
        f"## 📡 AVH 技術觀測日誌：{target_file}\n"
        f"* **觀測時間戳（CST）**：{timestamp}\n* **高維算力引擎**：{meta['llm_model']}\n\n---\n"
        f"### 1. 🌌 絕對本體觀測（Absolute Ontology）\n"
        f"* 🛡️ **本體論絕對指紋（Ontology Hex）**：[{data['user_hex']}] - **{data['state_name']}**\n"
        f"* **本體核心宣告（Core Statement）**：{meta['core_statement']}\n\n"
        f"**學術指紋（Academic Fingerprint）**：\n> **[基底狀態]** {data['state_desc']}\n>\n> **[演化觀測]** {meta['academic_fingerprint']}\n\n"
        f"**詳細本體量化儀表板（Ontology Quantification Dashboard）**：\n\n" + "\n\n".join(meta["user_dimension_logs"]) + "\n\n---\n"
        f"### 2. 🎣 背景能勢打撈（Background Field Retrieval）\n"
        f"* **場域建構狀態（Field Status）**：{meta.get('baseline_status', 'Void')}\n"
        f"* **重排日誌（Re-ranking Log）**：_{meta.get('filtering_log', 'N/A')}_\n"
        f"* **背景量化摘要**：_{meta.get('background_batch_log', 'N/A')}_\n"
        f"* **參考鄰域節點（Reference Neighborhood）**：\n" + "\n".join(meta.get("paper_records", [])) + "\n\n---\n"
        f"### 3. 📐 向量干涉量化（Quantified Vector Interference）\n"
        f"* **背景絕對指紋（Background Hex）**：[{data['baseline_hex']}] | 關係：**{meta.get('global_relation', 'N/A')}** | 相位角：{meta.get('global_angle', 'N/A')}\n"
        f"* **語意相近度**：{meta.get('global_proximity', 'N/A')} / 100\n\n" + "\n\n".join(meta.get("vector_logs", [])) + "\n\n---\n"
        f"### 4. 🧾 系統導讀\n> {data['summary']}\n\n"
    )

if __name__ == "__main__":
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files: sys.exit(0)
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V36.4 限流防禦與降維防爆日誌\n---\n")
        last_hex = ""
        for target in source_files:
            res = process_avh_manifestation(target)
            if res:
                last_hex = res["user_hex"]
                log_file.write(generate_trajectory_log(target, res))
            else:
                log_file.write(f"\n> ⚠️ {target} 掃描失敗或略過，詳見系統執行日誌。\n---\n")
    if last_hex:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a", encoding="utf-8") as env: env.write(f"HEX_CODE={last_hex}\n")
