import os
import sys
import json
import glob
import re
import requests
import urllib.parse
import time
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V33.0 幾何干涉與動態判準版 - 可審計角度與原點觀測)
# ==============================================================================

LLM_MODEL_NAME = 'openai/gpt-4o'
MD_FENCE = "`" * 3

print(f"🧠 [載入觀測核心] 啟動 V33.0 高維度大腦矩陣 ({LLM_MODEL_NAME})...")

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
                print(f"工具調用失敗，原因為 雲端算力請求超時或阻擋 ({e})")
                sys.exit(1)
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

def evaluate_user_text_and_compress(raw_text, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論觀測儀器」。請閱讀文本並評估 6 個維度。
維度定義：{manifest_str}
(注意：1=離群突破/sin, 0=合群守成/cos)

任務一：請根據文本內容，為這 6 個維度判定狀態 (sin 或是 cos)，並附上客觀的觀測判定短語。
任務二：為了向外部圖譜發動精準的語意檢索，請將這篇文本的最核心學術貢獻，壓縮成一句「極度精準的英文學術核心宣告 (Core Statement)」。
規則：長度 10 到 15 個單字，直指理論本質，提取最具原創性的物理/治理/拓樸概念，拒絕空泛八股。

請嚴格回傳以下 JSON 格式：
{MD_FENCE}json
{{
  "hex_code": "111000",
  "dim_logs": [
    "* **價值意圖**：離群突破 (sin) `[觀測判定：...]`",
    "* **治理維度**：離群突破 (sin) `[觀測判定：...]`",
    "* **認知深度**：離群突破 (sin) `[觀測判定：...]`",
    "* **描述架構**：合群守成 (cos) `[觀測判定：...]`",
    "* **擴張潛力**：合群守成 (cos) `[觀測判定：...]`",
    "* **應用實相**：合群守成 (cos) `[觀測判定：...]`"
  ],
  "core_statement": "Semantic topology and active damping for anti-fragile knowledge systems"
}}
{MD_FENCE}
"""
    print("🕸️ [大腦運算 - 階段 1] 測量本體絕對指紋，執行維度動態判定與核心宣告提取...")
    response = call_llm_with_retry(
        client, 
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:4000]}],
        temperature=0.1
    )
    return parse_llm_json(response.choices[0].message.content)

def fetch_broad_neighborhood_crossref(core_statement):
    headers = {
        "User-Agent": "AVH-Hologram-Engine/33.0 (https://github.com/alaric-kuo; mailto:open-source-bot@example.com)"
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
            print(f"⚠️ 核心宣告在 Crossref 查無任何文獻，將直接進入絕對無人區狀態。")
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
        print(f"工具調用失敗，原因為 Crossref 連線異常或超時 ({e})")
        sys.exit(1)

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
  "selected_ids": ["id_1", "id_2"],
  "filtering_log": "簡述保留了哪些文獻作為背景能勢參考，為何剔除其他雜訊"
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
    
    selected_ids = set(res.get("selected_ids", []))
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
【計算審計基準】：角度 (θ) 的判定必須基於『語意偏移權重』：
- 同向 (0~89度)：本質目標一致，僅在深度或擴張度有落差。
- 正交 (90度)：關注截然不同的系統切面，不構成直接阻力。
- 反向 (91~180度)：意圖修補舊典範，對本理論的離群演化構成拉扯與阻力。

維度定義：{manifest_str}

請嚴格回傳 JSON：
{MD_FENCE}json
{{
  "baseline_hex": "010011",
  "audit_formula": "Angle (θ) = f(本體維度判定, 背景維度判定, 語義偏移權重)",
  "global_angle": "整體相位差：105度 (偏向正交與反向阻力)",
  "vector_analysis": [
    {{"dimension": "價值意圖", "direction": "反向", "angle": "120度", "reason": "背景文獻尋求修補舊系統...與本理論重構邊界的意圖產生反向拉扯"}},
    {{"dimension": "治理維度", "direction": "同向", "angle": "45度", "reason": "雙方皆認同動態卸力...但本理論推演更深"}}
  ]
}}
{MD_FENCE}
"""
    print("📐 [場域測量 - 階段 4] 啟動反向尺規！以本體為原點，計算參考背景能勢之相位差與幾何干涉...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
        temperature=0.1
    )
    res = parse_llm_json(response.choices[0].message.content)
    return res

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text.strip()) < 100:
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
            paper_records.append("- `[Void]` **全域寂靜**：核心宣告過於前沿，目前打撈到之鄰近節點尚不足以構成可測量的背景能勢場。")
            vector_logs = ["* **測量結果**：周遭無足夠背景能勢質量，無法產生穩定的向量干涉與相位角。"]
            global_angle = "整體相位差：無定義 (Void)"
            audit_formula = "Angle (θ) = N/A (Insufficient Data)"
        else:
            baseline_status = f"Background Field Established (參考背景能勢建構完成：{len(final_papers)} 鄰域節點)"
            for p in final_papers:
                doi_link = f"https://doi.org/{p['id']}" if p['id'] != "Unknown" else "#"
                paper_records.append(f"- [[DOI 連結]({doi_link})] **{p['title']}**")
                
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
            messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": raw_text[:3000]}],
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
        print(f"工具調用失敗，原因為 處理管線執行異常 ({e})")
        sys.exit(1)

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
        f"> *註：本報告採 V33.0 原點觀測架構。將本體視為絕對坐標原點，以可審計之幾何角度測量其與外部學界之相位干涉。*\n"
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
        "        <h3>📡 學術價值全像儀 (AVH) 幾何干涉認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{data['summary']}</p>\n"
        "        <hr>\n"
        f"        <p>場域建構狀態：{meta['baseline_status']}</p>\n"
        f"        <p><strong>干涉審計公式：{meta['audit_formula']}</strong></p>\n"
        f"        <p><strong>整體場域偏差：{meta['global_angle']}</strong></p>\n"
        f"        <p>最終本體狀態：[ {data['user_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p><strong>學術指紋：</strong><br>{data['state_desc']}</p>\n"
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
        f"學術指紋：{data['state_desc']}\n\n"
        f"干涉審計公式：{data['meta_data']['audit_formula']}\n\n"
        f"整體場域偏差：{data['meta_data']['global_angle']}\n"
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
        log_file.write("# 📡 AVH 學術價值全像儀：V33.0 幾何干涉觀測日誌\n---\n")
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
