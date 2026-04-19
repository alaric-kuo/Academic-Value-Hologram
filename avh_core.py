import os
import sys
import json
import glob
import re
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import time
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V27.1 聯集能勢場 - 摘要減壓與延長射程版)
# ==============================================================================

LLM_MODEL_NAME = 'openai/gpt-4o'
MD_FENCE = "`" * 3

print(f"🧠 [載入觀測核心] 啟動 V27.1 高維度大腦矩陣 ({LLM_MODEL_NAME})...")

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN")
        sys.exit(1)
    return OpenAI(base_url="https://models.github.ai/inference", api_key=token)

def call_llm_with_retry(client, messages, temperature=0.1, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=LLM_MODEL_NAME, 
                temperature=temperature
            )
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"⚠️ 雲端連線異常 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試... [{e}]")
            if attempt == max_retries - 1:
                raise e
            time.sleep(wait_time)

def parse_llm_json(response_text):
    try:
        text = response_text.strip()
        pattern = f"{MD_FENCE}(?:json)?\\s*(\\{{.*?\\}})\\s*{MD_FENCE}"
        fence_match = re.search(pattern, text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))

        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(1))
        raise ValueError("找不到可解析的 JSON 區塊")
    except Exception as e:
        print(f"工具調用失敗，原因為 LLM JSON 解析失敗 ({e})")
        sys.exit(1)

def fetch_field_papers_union(keywords, limit=8):
    """【V27.1 核心】使用 OR 聯集查詢，限縮摘要範圍並延長等待，打穿 arXiv 防火牆"""
    headers = {"User-Agent": "AVH-Hologram/27.1 (GitHub Actions)"}
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    # 💥 從全域 (all) 降階為摘要 (abs)，減輕 arXiv 伺服器運算負擔
    search_query = " OR ".join([f"abs:{kw}" for kw in keywords])
    encoded_query = urllib.parse.quote(search_query)
    
    url = (f"https://export.arxiv.org/api/query?"
           f"search_query={encoded_query}&start=0&max_results={limit}"
           f"&sortBy=submittedDate&sortOrder=descending")
    
    print(f"🌍 [場域建構] 發射全域聯集探測網 (限縮摘要範圍，延長連線耐受度)...")
    field_papers = []
    
    for attempt in range(3):
        try:
            # 💥 Timeout 延長到 45 秒
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code in [403, 429]:
                print(f"⚠️ 觸發 arXiv 限流 (HTTP {response.status_code})，退避 10 秒...")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            for entry in root.findall('atom:entry', namespace):
                title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
                abstract = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
                paper_id = entry.find('atom:id', namespace).text.split('/')[-1]
                field_papers.append({
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract
                })
            break # 成功抓取整批，跳出重試
        except Exception as e:
            if attempt == 2:
                print(f"⚠️ 聯集探測網徹底失去連線 ({e})")
            time.sleep(5)
            
    return field_papers, search_query

def evaluate_user_text(raw_text, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論觀測儀器」。請閱讀文本並評估 6 個維度(1=突破, 0=守成)。
維度定義：{manifest_str}

【V27 核心任務】：為了建構「背景能勢場」，請根據文本內容，精準提煉出 8 個「完全不同領域或維度」的英文學術名詞(單字)。
這 8 個字必須涵蓋哲學、系統、治理、數學、資訊等廣泛的學術頻段(例如 Topology, Epistemology, Governance, Entropy, Kinematics 等)。

請回傳 JSON：
{MD_FENCE}json
{{
  "hex_code": "111111",
  "dim_logs": [
    "* **價值意圖**：離群突破 (sin) `[觀測判定：...]`",
    "* **治理維度**：離群突破 (sin) `[觀測判定：...]`",
    "* **認知深度**：離群突破 (sin) `[觀測判定：...]`",
    "* **描述架構**：離群突破 (sin) `[觀測判定：...]`",
    "* **擴張潛力**：離群突破 (sin) `[觀測判定：...]`",
    "* **應用實相**：離群突破 (sin) `[觀測判定：...]`"
  ],
  "field_vectors": ["Vector1", "Vector2", "Vector3", "Vector4", "Vector5", "Vector6", "Vector7", "Vector8"]
}}
{MD_FENCE}
"""
    print("🕸️ [大腦運算] 測量本體絕對指紋，並萃取八極背景向量...")
    try:
        response = call_llm_with_retry(
            client, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:4000]}],
            temperature=0.1
        )
        return parse_llm_json(response.choices[0].message.content)
    except Exception as e:
        print(f"工具調用失敗，原因為 GPT-4o 評估文本失敗 ({e})")
        sys.exit(1)

def evaluate_baseline_papers(papers, manifest):
    if not papers:
        return "000000", [0]*6
        
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    papers_str = json.dumps([{"title": p["title"], "abstract": p["abstract"]} for p in papers])
    
    sys_prompt = f"""
你正在測量當代學術的「背景能勢場」。以下是由多個不同領域關鍵字聯合抓出的前沿論文。
請綜合判斷這個論文群體所構成的場域，在 6 個維度上的整體表現。
維度定義：{manifest_str} (1=突破, 0=守成)

請回傳 JSON：
{MD_FENCE}json
{{
  "baseline_hex": "010011",
  "vote_stats": [2, 3, 1, 4, 5, 4] 
}}
{MD_FENCE}
"""
    print("⚖️ [場域測量] 計算背景能勢場之絕對張量...")
    try:
        response = call_llm_with_retry(
            client,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
            temperature=0.1
        )
        res = parse_llm_json(response.choices[0].message.content)
        return res.get("baseline_hex", "000000"), res.get("vote_stats", [0]*6)
    except Exception as e:
        print(f"工具調用失敗，原因為 GPT-4o 測量場域失敗 ({e})")
        sys.exit(1)

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text.strip()) < 100:
            return None

        # 1. User Hex & 8 Vectors
        user_data = evaluate_user_text(raw_text, manifest)
        user_hex = user_data.get("hex_code", "000000")
        dim_logs = user_data.get("dim_logs", [])
        field_vectors = user_data.get("field_vectors", [])[:8]
        user_state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        # 2. 建立背景能勢場 (聯集一波帶走)
        field_papers, query_str = fetch_field_papers_union(field_vectors)
        paper_records = []
        
        if len(field_papers) >= 4:
            baseline_status = f"Field Established (能勢場建構完成：{len(field_papers)} 節點)"
            baseline_hex, vote_stats = evaluate_baseline_papers(field_papers, manifest)
            for p in field_papers:
                paper_records.append(f"- `[{p['id']}]` *{p['title']}*")
        else:
            baseline_status = "Field Fracture (能勢場建構破裂：節點不足)"
            baseline_hex = "000000"
            vote_stats = [0]*6
            paper_records.append("- (探測網遭阻擋或失效，無法建立穩定能勢場)")

        # 3. 計算三角校正偏移值 (Offset)
        breakthrough_dims = []
        offset_score = 0
        for i in range(6):
            if user_hex[i] == "1" and baseline_hex[i] == "0":
                breakthrough_dims.append(manifest["dimensions"][list(manifest["dimensions"].keys())[i]]["layer"])
                offset_score += 1
            elif user_hex[i] == "0" and baseline_hex[i] == "1":
                offset_score -= 1
                
        breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與場域同頻"
        offset_status = f"能勢偏移值 (Offset): {offset_score:+d} (正值為突破，負值為受迫)"
        
        # 4. 導讀摘要
        client = get_llm_client()
        summary_prompt = f"""
本理論在「背景能勢場」中測得偏移值為 {offset_score:+d}，突破維度：【{breakthrough_str}】。
請根據下文撰寫 200 字理論導讀，客觀描述其在學術場域中的相對定位與運作邏輯。
第一句必須以「本理論架構...」開頭。
"""
        response = call_llm_with_retry(
            client,
            messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": raw_text[:3000]}],
            temperature=0.2
        )
        clean_summary = zhconv.convert(response.choices[0].message.content.strip(), 'zh-tw')

        return {
            "user_hex": user_hex,
            "baseline_hex": baseline_hex,
            "breakthrough_str": breakthrough_str,
            "offset_status": offset_status,
            "state_name": user_state_info['name'],
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "field_vectors": field_vectors,
                "arxiv_query": query_str,
                "valid_hits": len(field_papers),
                "paper_records": paper_records,
                "vote_stats": vote_stats,
                "baseline_status": baseline_status,
                "llm_model": LLM_MODEL_NAME
            }
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 處理管線異常 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])
    meta = data['meta_data']
    papers_text = "\n".join(meta['paper_records'])
    vector_str = ", ".join(meta['field_vectors'])
    
    if "Established" in meta['baseline_status']:
        vote_str = " | ".join([f"Dim{i+1}: {meta['vote_stats'][i]}/{meta['valid_hits']}" for i in range(6)])
    else:
        vote_str = "場域破裂，無張量數據"

    log_output = (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳 (CST)**：`{timestamp}`\n"
        f"* **高維算力引擎**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 背景能勢場建立 (Background Energy-Potential Field)\n"
        f"* **全域聯集陣列**：`{meta['arxiv_query']}`\n"
        f"* **場域建構狀態**：`{meta['baseline_status']}`\n"
        f"* **能勢場代表節點 (Top Articles)**：\n"
        f"{papers_text}\n\n"
        f"* **場域張量統計**：`[ {vote_str} ]`\n"
        f"* 🗺️ **背景絕對指紋 (Background Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"### 2. ⚖️ 三角校正與能勢干涉 (Triangulation & Interference)\n"
        f"* 🛡️ **本體論絕對指紋 (Ontology Hex)**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* ⚡ **{data['offset_status']}**\n"
        f"* 🌌 **拓樸破缺維度 (相對於背景場)**：**【{data['breakthrough_str']}】**\n\n"
        f"**詳細本體測量儀表板**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"> *註：本報告採 V27.1 全域聯集能勢場干涉測量。*\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    html_content = data['full_text'].replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = data['meta_data']
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {html_content}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <h3>📡 學術價值全像儀 (AVH) 背景場域認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{data['summary']}</p>\n"
        "        <hr>\n"
        f"        <p>場域建構狀態：{meta['baseline_status']}</p>\n"
        f"        <p><strong>{data['offset_status']}</strong></p>\n"
        f"        <p>突破維度：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終本體狀態：[ {data['user_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open(f"WP_Ready_{basename}.html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    tex_content = data['full_text'].replace("#", "\\section")
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        f"\\title{{{basename}}}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        f"[{data['user_hex']}] {data['state_name']}。\n\n"
        f"{data['offset_status']}\n"
        "\\end{abstract}\n\n"
        f"{tex_content}\n\n"
        "\\end{document}\n"
    )
    with open(f"{basename}_Archive.tex", "w", encoding="utf-8") as f:
        f.write(tex_output)

if __name__ == "__main__":
    if not os.path.exists("avh_manifest.json"):
        print("工具調用失敗，原因為 遺失底層定義檔")
        sys.exit(1)
        
    with open("avh_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files:
        sys.exit(0)
        
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V27.1 聯集能勢場觀測日誌\n---\n")
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['user_hex']
                log_file.write(generate_trajectory_log(target_source, result_data))
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data) 

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
