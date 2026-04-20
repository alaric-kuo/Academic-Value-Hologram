import os
import sys
import json
import glob
import re
import requests
import urllib.parse
import time
import html
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V33.2 結構防禦與批次穩定版 - 容錯隔離與 Schema 審計)
# ==============================================================================

LLM_MODEL_NAME = 'openai/gpt-4o'
MD_FENCE = "`" * 3

print(f"🧠 [載入觀測核心] 啟動 V33.2 高維度大腦矩陣 ({LLM_MODEL_NAME})...")

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("遺失 GITHUB_TOKEN，無法啟動算力。")
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
                raise ConnectionError(f"雲端算力請求超時或阻擋 ({e})")
            time.sleep(wait_time)

def parse_llm_json(response_text):
    try:
        text = response_text.strip()
        # V33.2 修正：將正則表達式改為非貪婪模式 (.*?)，防止吞噬多餘字元
        pattern = f"{MD_FENCE}(?:json)?\\s*(\\{{.*?\\}})\\s*{MD_FENCE}"
        fence_match = re.search(pattern, text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))

        obj_match = re.search(r"(\{.*?\})", text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(1))
        raise ValueError("找不到可解析的 JSON 區塊")
    except Exception as e:
        raise ValueError(f"LLM JSON 解析失敗 ({e})\n原始輸出片段：{response_text[:200]}...")

def evaluate_user_text_and_compress(raw_text, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論觀測儀器」。請閱讀文本並評估 6 個維度。
維度定義：{manifest_str}
(注意：1=離群突破/sin, 0=合群守成/cos)

任務一：請根據文本內容，為這 6 個維度進行獨立判定 (sin 或是 cos)，不可照抄範例，必須動態計算。
任務二：請將這篇文本的最核心學術貢獻，壓縮成一句「極度精準的英文學術核心宣告 (Core Statement)」。(長度 10-15 字，拒絕八股)

請嚴格回傳以下 JSON 格式 (角括號內為必須由你動態填寫的變數)：
{MD_FENCE}json
{{
  "hex_code": "<在此填寫判定後的 6 位數二進制字串，例如 101010>",
  "dim_logs": [
    "* **價值意圖**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`",
    "* **治理維度**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`",
    "* **認知深度**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`",
    "* **描述架構**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`",
    "* **擴張潛力**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`",
    "* **應用實相**：<在此填寫: 離群突破 (sin) 或 合群守成 (cos)> `[觀測判定：<撰寫客觀觀測短語>]`"
  ],
  "core_statement": "<在此填寫 10-15 字的核心宣告>"
}}
{MD_FENCE}
"""
    print("🕸️ [大腦運算 - 階段 1] 測量本體絕對指紋，強制啟動動態判定...")
    # V33.2 修正：擴大讀取範圍以涵蓋後段論述，並將核心判定降溫至 0.0 以降低漂移
    response = call_llm_with_retry(
        client, 
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:8000]}],
        temperature=0.0 
    )
    res = parse_llm_json(response.choices[0].message.content)
    
    # Schema 驗證
    if not re.match(r"^[01]{6}$", res.get("hex_code", "")):
        raise ValueError(f"Hex Code 格式異常：{res.get('hex_code')}")
    if len(res.get("dim_logs", [])) != 6:
        raise ValueError(f"維度日誌數量異常：需為 6，取得 {len(res.get('dim_logs', []))}")
        
    return res

def fetch_broad_neighborhood_crossref(core_statement):
    headers = {
        "User-Agent": "AVH-Hologram-Engine/33.2 (https://github.com/alaric-kuo; mailto:open-source-bot@example.com)"
    }
    encoded_query = urllib.parse.quote(core_statement)
    url = f"https://api.crossref.org/works?query={encoded_query}&select=DOI,title,abstract&rows=25"
    
    print(f"🌍 [實體觀測 - 階段 2] 投放核心宣告：『{core_statement}』\n🌍 正在 Crossref 禮貌池中打撈關聯文獻...")
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 429:
            print(f"⚠️ 遭遇 Crossref 瞬間限流，強制退避 5 秒...")
            time.sleep(5)
            response = requests.get(url, headers=headers, timeout=20)
            
        response.raise_for_status()
        data = response.json()
        
        items = data.get("message", {}).get("items", [])
        raw_papers = []
        
        if not items:
            return raw_papers
            
        for paper in items:
            raw_abstract = paper.get("abstract")
            if not raw_abstract: 
                continue
            clean_abstract = re.sub(r'<[^>]+>', '', raw_abstract) 
            title = paper.get("title", [""])[0] if paper.get("title") else "Unknown"
            
            raw_papers.append({
                "id": paper.get("DOI", "Unknown"),
                "title": title,
                "abstract": clean_abstract[:600]
            })
            
            if len(raw_papers) >= 20:
                break
                
        print(f"🌍 成功撈取 {len(raw_papers)} 篇具備摘要之文獻，準備進行參考背景重排...")
        time.sleep(1)
        return raw_papers
        
    except Exception as e:
        raise ConnectionError(f"Crossref 連線異常或超時 ({e})")

def rerank_and_filter_papers(core_statement, raw_papers):
    if not raw_papers:
        return [], "無可用文獻進行重排。"
        
    client = get_llm_client()
    papers_json = json.dumps(raw_papers, ensure_ascii=False)
    
    sys_prompt = f"""
你現在是一位客觀的學術觀測員。本理論核心宣告為："{core_statement}"
以下是傳統搜尋引擎撈回的 {len(raw_papers)} 篇初步文獻。

請利用高維度認知閱讀它們。剔除「只是撞字、核心邏輯毫無關聯」的雜訊。
挑出「最適合拿來作為該理論『參考背景座標』或『對話對象』」的文獻 (最多保留 8 篇，如果只有 1 篇合格就留 1 篇，0 篇則回傳空陣列)。

請回傳 JSON：
{MD_FENCE}json
{{
  "selected_ids": ["<填寫保留的 id>"],
  "filtering_log": "<簡述保留了哪些文獻作為背景能勢參考，為何剔除其他雜訊>"
}}
{MD_FENCE}
"""
    print(f"⚖️ [大腦運算 - 階段 3] 啟動柔性重排，萃取符合條件的參考背景能勢...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_json}],
        temperature=0.0
    )
    res = parse_llm_json(response.choices[0].message.content)
    
    # Schema 驗證
    raw_selected = res.get("selected_ids", [])
    if not isinstance(raw_selected, list):
        raise TypeError("selected_ids 必須是陣列 (list)")
        
    selected_ids = set([str(sid) for sid in raw_selected])
    filtering_log = res.get("filtering_log", "執行標準過濾機制。")
    
    final_papers = [p for p in raw_papers if p["id"] in selected_ids][:8]
    return final_papers, filtering_log

def evaluate_matrix_with_reverse_ruler(papers, manifest, core_statement):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    papers_str = json.dumps([{"title": p["title"], "abstract": p["abstract"]} for p in papers])
    
    sys_prompt = f"""
你現在是一台「幾何干涉與相位測量儀」。
【觀測原點】：本理論核心為 "{core_statement}"。請將此視為向量空間的「絕對原點 (0度)」。
以下是由系統篩選出、目前學界最接近的 {len(papers)} 篇參考背景文獻。

請以本理論為尺規，測量這些背景研究的相位角。
【計算審計基準】：角度 (θ) 的判定必須基於『語意偏移權重』，必須動態計算，不可照抄範例：
- 同向 (0~89度)：本質目標一致，僅在深度或擴張度有落差。
- 正交 (90度)：關注截然不同的系統切面，不構成直接阻力。
- 反向 (91~180度)：意圖修補舊典範，對本理論的離群演化構成拉扯與阻力。

維度定義：{manifest_str}

請嚴格回傳 JSON (角括號內為強制動態計算的變數)：
{MD_FENCE}json
{{
  "baseline_hex": "<動態計算 6 位數字串>",
  "audit_formula": "Angle (θ) = f(本體維度判定, 背景維度判定, 語義偏移權重)",
  "global_angle": "整體相位差：<動態計算之角度>度 (<動態判定之干涉狀態>)",
  "vector_analysis": [
    {{"dimension": "<維度名稱>", "direction": "<動態判定: 同向/正交/反向>", "angle": "<動態計算之角度>度", "reason": "<動態論述原因>"}}
  ]
}}
{MD_FENCE}
"""
    print("📐 [場域測量 - 階段 4] 啟動反向尺規！強制大腦實時計算干涉角度...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
        temperature=0.0
    )
    res = parse_llm_json(response.choices[0].message.content)
    
    # Schema 驗證
    if not re.match(r"^[01]{6}$", res.get("baseline_hex", "")):
        raise ValueError(f"背景 Hex Code 格式異常：{res.get('baseline_hex')}")
    vectors = res.get("vector_analysis", [])
    if len(vectors) != 6:
        raise ValueError(f"向量分析數量異常：需為 6 維度，取得 {len(vectors)}")
        
    return res

def escape_latex(text):
    """V33.2 新增：基礎 LaTeX 特殊字元跳脫"""
    chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', 
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', 
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    # 先跳脫反斜線，避免重複跳脫
    text = text.replace('\\', chars['\\'])
    for k, v in chars.items():
        if k != '\\':
            text = text.replace(k, v)
    return text

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text.strip()) < 100:
            print("⚠️ 文本過短，略過掃描。")
            return None

        user_data = evaluate_user_text_and_compress(raw_text, manifest)
        user_hex = user_data["hex_code"]
        dim_logs = user_data["dim_logs"]
        core_statement = user_data.get("core_statement", "Academic Ontology Theory")
        
        state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        user_state_name = state_info["name"]
        user_state_desc = state_info["desc"]
        
        raw_papers = fetch_broad_neighborhood_crossref(core_statement)
        final_papers, filtering_log = rerank_and_filter_papers(core_statement, raw_papers)
        
        paper_records = []
        vector_logs = []
        global_angle = ""
        audit_formula = ""
        
        if not final_papers:
            baseline_status = "Sparse Reference Field (稀疏參考場：打撈數量不足以建構穩定母體)"
            baseline_hex = "000000"
            paper_records.append("- `[Void]` **全域寂靜**：核心宣告目前打撈到之鄰近節點尚不足以構成可測量的背景能勢場。")
            vector_logs = ["* **測量結果**：周遭無足夠背景能勢質量，無法產生穩定的向量干涉與相位角。"]
            global_angle = "整體相位差：無定義 (Void)"
            audit_formula = "Angle (θ) = N/A (Insufficient Data)"
        else:
            baseline_status = f"Background Field Established (參考背景能勢建構完成：{len(final_papers)} 鄰域節點)"
            for p in final_papers:
                doi_link = f"https://doi.org/{p['id']}" if p['id'] != "Unknown" else "#"
                # V33.2 修正：標準 Markdown 連結格式
                paper_records.append(f"- [DOI 連結]({doi_link}) **{p['title']}**")
                
            matrix_data = evaluate_matrix_with_reverse_ruler(final_papers, manifest, core_statement)
            baseline_hex = matrix_data.get("baseline_hex", "000000")
            global_angle = matrix_data.get("global_angle", "未定義")
            audit_formula = matrix_data.get("audit_formula", "Angle (θ) = f(ΔV, ΔC)")
            
            for v in matrix_data.get("vector_analysis", []):
                vector_logs.append(f"* **{v['dimension']}**：【{v['direction']}】(偏角 {v['angle']}) - {v['reason']}")

        client = get_llm_client()
        summary_prompt = f"""
本理論在「外部場域觀測」中，與現有參考背景能勢的相對位置為：{global_angle}。
請根據下文撰寫 200 字理論導讀，客觀描述其作為觀測原點，是如何與現有學界產生干涉與拉扯的。若是無人區請直接指出。
第一句必須以「本理論架構...」開頭。
"""
        response = call_llm_with_retry(
            client,
            messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": raw_text[:4000]}],
            temperature=0.2
        )
        clean_summary = zhconv.convert(response.choices[0].message.content.strip(), 'zh-tw')

        return {
            "user_hex": user_hex,
            "baseline_hex": baseline_hex,
            "state_name": user_state_name,
            "state_desc": user_state_desc,
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "core_statement": core_statement,
                "raw_hits": len(raw_papers),
                "final_hits": len(final_papers),
                "filtering_log": filtering_log,
                "paper_records": paper_records,
                "vector_logs": vector_logs,
                "global_angle": global_angle,
                "audit_formula": audit_formula,
                "baseline_status": baseline_status,
                "llm_model": LLM_MODEL_NAME
            }
        }
    except Exception as e:
        # V33.2 修正：捕獲異常但不中斷程式，確保批次執行能繼續
        print(f"❌ 檔案 {source_path} 處理失敗: {e}")
        return None

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n\n".join(data['dim_logs'])
    meta = data['meta_data']
    papers_text = "\n".join(meta['paper_records'])
    vectors_text = "\n\n".join(meta['vector_logs'])
    
    log_output = (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳 (CST)**：`{timestamp}`\n"
        f"* **高維算力引擎**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 絕對本體觀測 (Absolute Ontology)\n"
        f"* 🛡️ **本體論絕對指紋 (Ontology Hex)**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* **本體核心宣告 (Core Statement)**：`{meta['core_statement']}`\n\n"
        f"**學術指紋 (Academic Fingerprint)**：\n"
        f"> {data['state_desc']}\n\n"
        f"**詳細本體測量儀表板**：\n\n"
        f"{dim_logs_text}\n\n"
        f"---\n"
        f"### 2. 🎣 背景能勢打撈 (Background Field Retrieval)\n"
        f"* **場域建構狀態**：`{meta['baseline_status']}` (原始打撈 {meta['raw_hits']} 篇)\n"
        f"* **大腦重排日誌 (Re-ranking Filter)**：_{meta['filtering_log']}_\n"
        f"* **參考鄰域節點 (Reference Neighborhood)**：\n"
        f"{papers_text}\n\n"
        f"---\n"
        f"### 3. 📐 反向尺規：幾何干涉與相位角 (Geometric Interference)\n"
        f"> *以本理論為絕對觀測原點(0度)，測量現有背景能勢之發展向量*\n\n"
        f"* 🧮 **干涉審計公式**：`{meta['audit_formula']}`\n"
        f"* 🌐 **整體場域偏差**：**{meta['global_angle']}**\n"
        f"* 🗺️ **背景絕對指紋 (Background Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"**維度向量干涉儀表板**：\n\n"
        f"{vectors_text}\n\n"
        f"---\n"
        f"> *註：本報告採 V33.2 結構防禦版。實裝 Schema 審計與批次隔離容錯，確保觀測穩定性。*\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    # V33.2 修正：加入 HTML 轉義防護
    safe_full_text = html.escape(data['full_text']).replace('\n', '<br>')
    safe_summary = html.escape(data['summary'])
    safe_desc = html.escape(data['state_desc'])
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = data['meta_data']
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {safe_full_text}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <h3>📡 學術價值全像儀 (AVH) 幾何干涉認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{safe_summary}</p>\n"
        "        <hr>\n"
        f"        <p>場域建構狀態：{meta['baseline_status']}</p>\n"
        f"        <p><strong>干涉審計公式：{html.escape(meta['audit_formula'])}</strong></p>\n"
        f"        <p><strong>整體場域偏差：{html.escape(meta['global_angle'])}</strong></p>\n"
        f"        <p>最終本體狀態：[ {data['user_hex']} ] - <strong>{html.escape(data['state_name'])}</strong></p>\n"
        f"        <p><strong>學術指紋：</strong><br>{safe_desc}</p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open(f"WP_Ready_{basename}.html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    # V33.2 修正：實裝 LaTeX 轉義防護，並保留標題轉換
    safe_text = escape_latex(data['full_text']).replace(r"\#", "\\section")
    safe_desc = escape_latex(data['state_desc'])
    meta = data['meta_data']
    
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
        f"[{data['user_hex']}] {escape_latex(data['state_name'])}。\n\n"
        f"學術指紋：{safe_desc}\n\n"
        f"干涉審計公式：{escape_latex(meta['audit_formula'])}\n\n"
        f"整體場域偏差：{escape_latex(meta['global_angle'])}\n"
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
        
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V33.2 批次穩定觀測日誌\n---\n")
        last_hex_code = ""
        
        # V33.2 修正：Per-file 隔離容錯
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['user_hex']
                log_file.write(generate_trajectory_log(target_source, result_data))
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data) 
            else:
                log_file.write(f"\n> ⚠️ `[{target_source}]` 掃描失敗或略過，詳見系統執行日誌。\n---\n")

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
