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
# AVH Genesis Engine (V25.2 算力解放與絕對抗阻版 - 指數退避防禦)
# ==============================================================================

LLM_MODEL_NAME = 'openai/gpt-4o'
MD_FENCE = "`" * 3

print(f"🧠 [載入觀測核心] 啟動 V25 高維度大腦矩陣 ({LLM_MODEL_NAME})...")

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN")
        sys.exit(1)
    return OpenAI(base_url="https://models.github.ai/inference", api_key=token)

def call_llm_with_retry(client, messages, temperature=0.1, max_retries=4):
    """具備指數退避防禦的 LLM 呼叫引擎，專治 Connection Error 與 Rate Limit"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=LLM_MODEL_NAME, 
                temperature=temperature
            )
            return response
        except Exception as e:
            wait_time = 2 ** attempt  # 1, 2, 4, 8 秒...
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

def fetch_arxiv_papers(query_terms, limit=10):
    search_query = "+AND+".join([f"all:{t}" for t in query_terms])
    encoded_query = urllib.parse.quote(search_query, safe=":+")
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&start=0&max_results={limit}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    headers = {"User-Agent": "AVH-Hologram/25.2 (GitHub Actions)"}
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            raw_papers = []
            for entry in root.findall('atom:entry', namespace):
                title_node = entry.find('atom:title', namespace)
                abstract_node = entry.find('atom:summary', namespace)
                id_node = entry.find('atom:id', namespace)
                
                if title_node is None or abstract_node is None:
                    continue
                title = title_node.text.strip().replace('\n', ' ')
                abstract = abstract_node.text.strip().replace('\n', ' ')
                paper_id = id_node.text.split('/')[-1] if id_node is not None else "Unknown"
                raw_papers.append({"id": paper_id, "title": title, "abstract": abstract})
                
            return raw_papers, search_query.replace("+AND+", " AND ")
        except Exception as e:
            print(f"⚠️ arXiv 連線異常 (嘗試 {attempt + 1}/3)... [{e}]")
            if attempt == 2:
                print(f"工具調用失敗，原因為 arXiv API 徹底失去連線 ({e})")
                sys.exit(1)
            time.sleep(2)

def evaluate_user_text(raw_text, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論觀測儀器」。
請閱讀使用者的文本，並根據以下 6 個維度進行評判。
若文本展現出「打破框架、重構邊界、元科學」的特質，該維度記為 "1" (sin)。
若文本屬於「既有框架內解題、常規工程」，該維度記為 "0" (cos)。
維度定義：{manifest_str}

另外，請從文本中提煉出 3 個最能代表其學術意圖的「英文單字 (名詞)」，用於檢索 arXiv (例如 Governance, Epistemic, Topology 等)。

請嚴格以下列 JSON 格式回傳，不要有任何多餘對話：
{MD_FENCE}json
{{
  "hex_code": "111111",
  "dim_logs": [
    "* **價值意圖**：離群突破 (sin) `[觀測判定：(簡短說明為何給1或0)]`",
    "* **治理維度**：離群突破 (sin) `[觀測判定：(簡短說明)]`",
    "* **認知深度**：離群突破 (sin) `[觀測判定：(簡短說明)]`",
    "* **描述架構**：離群突破 (sin) `[觀測判定：(簡短說明)]`",
    "* **擴張潛力**：離群突破 (sin) `[觀測判定：(簡短說明)]`",
    "* **應用實相**：離群突破 (sin) `[觀測判定：(簡短說明)]`"
  ],
  "arxiv_keywords": ["Keyword1", "Keyword2", "Keyword3"]
}}
{MD_FENCE}
"""
    print("🕸️ [大腦運算] GPT-4o 正在讀取文本並進行高維拓樸測量...")
    try:
        response = call_llm_with_retry(
            client, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:4000]}],
            temperature=0.1
        )
        return parse_llm_json(response.choices[0].message.content)
    except Exception as e:
        print(f"工具調用失敗，原因為 GPT-4o 評估文本失敗且重試耗盡 ({e})")
        sys.exit(1)

def evaluate_baseline_papers(papers, manifest):
    if not papers:
        return "000000", [0]*6
        
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    papers_str = json.dumps([{"title": p["title"], "abstract": p["abstract"]} for p in papers])
    
    sys_prompt = f"""
你是一台學術觀測儀器。請閱讀以下 arXiv 前沿論文摘要，判斷這個「論文群體」在 6 個維度上的綜合表現。
維度定義：{manifest_str}
1=突破框架(sin)，0=保守解題(cos)。多數決判定。

請回傳 JSON：
{MD_FENCE}json
{{
  "baseline_hex": "010011",
  "vote_stats": [0, 3, 0, 1, 4, 4] 
}}
{MD_FENCE}
"""
    try:
        response = call_llm_with_retry(
            client,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
            temperature=0.1
        )
        res = parse_llm_json(response.choices[0].message.content)
        return res.get("baseline_hex", "000000"), res.get("vote_stats", [0]*6)
    except Exception as e:
        print(f"工具調用失敗，原因為 GPT-4o 評估 arXiv 基準失敗且重試耗盡 ({e})")
        sys.exit(1)

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在處理實體源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text.strip()) < 100:
            return None

        # 1. 取得 User 絕對指紋與檢索詞
        user_data = evaluate_user_text(raw_text, manifest)
        
        user_hex = user_data["hex_code"]
        dim_logs = user_data["dim_logs"]
        search_terms = user_data["arxiv_keywords"]
        user_state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        # 2. arXiv 檢索與領域過濾
        print(f"🌍 [實體觀測] 啟動跨語系檢索，英文錨點：{search_terms}")
        raw_papers, actual_query = fetch_arxiv_papers(search_terms)
        
        domain_keywords = ["academic", "research", "evaluation", "governance", "knowledge", "scholarly", "trust", "epistemic", "system", "entropy", "topology"]
        valid_papers = []
        for p in raw_papers:
            text_to_check = (p['title'] + " " + p['abstract']).lower()
            if any(k in text_to_check for k in domain_keywords):
                valid_papers.append(p)
                if len(valid_papers) == 5:
                    break
        
        paper_records = []
        if len(valid_papers) >= 2:
            baseline_status = "Valid (高維領域命中)"
            print("⚖️ [基準對撞] GPT-4o 正在裁決 arXiv 論文群...")
            baseline_hex, vote_stats = evaluate_baseline_papers(valid_papers, manifest)
            for p in valid_papers:
                paper_records.append(f"- `[{p['id']}]` {p['title']}")
        else:
            baseline_status = "Observation Fracture (觀測破裂：查無同頻文獻)"
            baseline_hex = "000000"
            vote_stats = [0]*6
            paper_records.append(f"- (原始抓取 {len(raw_papers)} 篇，但均無法與本體論產生語意共振)")
            paper_records.append("- (系統坦承觀測破裂，無法建立有效外部基準)")

        # 3. 產出客觀導讀摘要
        breakthrough_dims = [manifest["dimensions"][list(manifest["dimensions"].keys())[i]]["layer"] 
                             for i in range(6) if user_hex[i] == "1" and baseline_hex[i] == "0"]
        if "Valid" not in baseline_status:
            breakthrough_str = "基準破裂，無法進行相對維度測量"
        else:
            breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與傳統基準同頻"
        
        client = get_llm_client()
        summary_prompt = f"""
你是一位專注於系統工程與物理本體論的客觀解讀者。
本理論經過實體文獻對比，測量狀態為：【{baseline_status}】。理論狀態定位為：{user_state_info['name']}。
請根據下文撰寫約 200 字的「理論導讀摘要」。語氣客觀，禁止使用宣傳式修辭。
第一句話必須以「本理論架構...」開頭。
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
            "state_name": user_state_info['name'],
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "arxiv_query": actual_query,
                "raw_hits": len(raw_papers),
                "valid_hits": len(valid_papers),
                "paper_records": paper_records,
                "vote_stats": vote_stats,
                "baseline_status": baseline_status,
                "llm_model": LLM_MODEL_NAME
            }
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 處理管線發生未預期錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])
    meta = data['meta_data']
    papers_text = "\n".join(meta['paper_records'])
    
    if "Valid" in meta['baseline_status']:
        vote_str = " | ".join([f"Dim{i+1}: {meta['vote_stats'][i]}/{meta['valid_hits']}" for i in range(6)])
    else:
        vote_str = "觀測破裂，無基準數據"

    log_output = (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳 (CST)**：`{timestamp}`\n"
        f"* **高維算力引擎**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🔬 觀測參數與基準建立 (Observation Parameters)\n"
        f"* **跨語系動態檢索 (Query)**：`{meta['arxiv_query']}`\n"
        f"* **基準校準狀態**：`{meta['baseline_status']}`\n"
        f"* **有效觀測樣本**：\n"
        f"{papers_text}\n\n"
        f"* **基準矩陣投票統計**：`[ {vote_str} ]`\n"
        f"* 🗺️ **外部真實基準 (Baseline Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"### 2. ⚖️ 高維差分干涉結果 (Interference Results)\n"
        f"* 🛡️ **本體論絕對指紋**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* 🌌 **拓樸破缺維度 (相對於有效基準)**：**【{data['breakthrough_str']}】**\n\n"
        f"**詳細大腦測量儀表板**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"> *註：本報告已全面升級為 GPT-4o 高維張量測量。*\n"
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
        "        <h3>📡 學術價值全像儀 (AVH) 高維觀測認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{data['summary']}</p>\n"
        "        <hr>\n"
        f"        <p>觀測基準狀態：{meta['baseline_status']}</p>\n"
        f"        <p>突破維度：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終演化狀態：[ {data['user_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
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
        f"觀測狀態：{data['meta_data']['baseline_status']}\n"
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
        log_file.write("# 📡 AVH 學術價值全像儀：V25 算力解放觀測日誌\n---\n")
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
