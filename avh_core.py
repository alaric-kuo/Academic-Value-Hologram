import os
import sys
import json
import glob
import numpy as np
import networkx as nx
import re
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V24.6 真實語意降階與破裂坦承版)
# ==============================================================================

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL_NAME = 'openai/gpt-4o'

print(f"🧠 [載入觀測核心] 正在啟動 IQD 物理差分矩陣 ({EMBEDDING_MODEL_NAME})...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"工具調用失敗，原因為 本地向量模型載入錯誤 ({e})")
    sys.exit(1)

def fetch_arxiv_papers(query_terms, limit=10):
    """底層 arXiv 呼叫引擎，不含任何 Hard Code"""
    search_query = "+AND+".join([f"all:{t}" for t in query_terms])
    encoded_query = urllib.parse.quote(search_query, safe=":+")
    
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&start=0&max_results={limit}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    headers = {"User-Agent": "AVH-Hologram/24.6 (GitHub Actions)"}
    
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
        print(f"⚠️ arXiv API 呼叫異常 ({e})")
        return [], "API Error"

def get_valid_baseline(extracted_words):
    """真實語意降階邏輯：拒絕塞詞，無對應則坦承破裂"""
    if len(extracted_words) < 2:
        return [], "無法從核心波包提取足夠的檢索維度", 0, "Observation Fracture (提取破裂)"

    domain_keywords = ["academic", "research", "evaluation", "governance", "knowledge", "scholarly", "trust", "epistemic", "system", "entropy", "topology"]
    
    # 【Tier 1：精準打擊】使用前 3 個字
    tier1_terms = extracted_words[:3]
    print(f"🌍 [實體觀測] Tier 1 精準檢索，關鍵字：{tier1_terms}...")
    raw_papers, actual_query = fetch_arxiv_papers(tier1_terms)
    
    valid_papers = []
    for p in raw_papers:
        text_to_check = (p['title'] + " " + p['abstract']).lower()
        if any(k in text_to_check for k in domain_keywords):
            valid_papers.append(p)
            if len(valid_papers) == 5:
                break
                
    if len(valid_papers) >= 2:
        return valid_papers, actual_query, len(raw_papers), "Valid (Tier 1 命中)"

    # 【Tier 2：次要語意降階】如果文章字數夠，使用第 2 到第 4 個字
    if len(extracted_words) >= 4:
        tier2_terms = extracted_words[1:4]
        print(f"⚠️ [降階觀測] Tier 1 查無實質對應，降階至次要語意：{tier2_terms}...")
        raw_papers, actual_query = fetch_arxiv_papers(tier2_terms)
        
        valid_papers = []
        for p in raw_papers:
            text_to_check = (p['title'] + " " + p['abstract']).lower()
            if any(k in text_to_check for k in domain_keywords):
                valid_papers.append(p)
                if len(valid_papers) == 5:
                    break
                    
        if len(valid_papers) >= 2:
            return valid_papers, actual_query, len(raw_papers), "Valid (Tier 2 降階命中)"

    # 【坦承破裂】如果還是沒有，絕對不塞 Hard Code
    return [], actual_query, len(raw_papers), "Observation Fracture (觀測破裂：查無同頻文獻)"

def call_copilot_brain(system_prompt, user_prompt):
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN")
        sys.exit(1)
        
    print(f"🧠 [雲端大腦] 正在連線 GitHub Models Inference ({LLM_MODEL_NAME})...")
    client = OpenAI(base_url="https://models.github.ai/inference", api_key=token)
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL_NAME, 
            temperature=0.2,
            max_tokens=800
        )
        return zhconv.convert(response.choices[0].message.content.strip(), 'zh-tw')
    except Exception as e:
        print(f"工具調用失敗，原因為 GitHub Models API 呼叫錯誤 ({e})")
        sys.exit(1)

def compute_iqd_hex(text_vec, manifest):
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    hex_code = ""
    dim_logs = []
    
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        
        sim_sin = float(np.dot(text_vec, v_sin) / (np.linalg.norm(text_vec) * np.linalg.norm(v_sin)))
        sim_cos = float(np.dot(text_vec, v_cos) / (np.linalg.norm(text_vec) * np.linalg.norm(v_cos)))
        
        diff = sim_sin - sim_cos
        bit = "1" if diff > 0.0 else "0" 
        hex_code += bit
        
        winner = "離群突破 (sin)" if bit == "1" else "守成合群 (cos)"
        dim_logs.append(f"* **{dim['layer']}**：{winner} `[Δ Diff: {diff:+.4f}]`")
        
    return hex_code, dim_logs

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            return None
            
        embeddings = embedding_model.encode(paragraphs)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        peak_embeddings = [embeddings[i] for i in ranked_indices[:3]]
        psi_peak = np.mean(peak_embeddings, axis=0) 
        top_sentence = paragraphs[ranked_indices[0]]

        print("🕸️ [IQD 差分] 計算絕對指紋...")
        user_hex, dim_logs = compute_iqd_hex(psi_peak, manifest)
        user_state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})

        # 擷取純文本字彙進行動態降階搜尋
        extracted_words = re.findall(r'\b[A-Za-z]{4,}\b', top_sentence)
        valid_papers, actual_query, raw_hits, baseline_status = get_valid_baseline(extracted_words)
        
        baseline_hex_votes = []
        vote_stats = [0]*6 
        paper_records = []
        
        if "Valid" in baseline_status:
            for p in valid_papers:
                p_vec = embedding_model.encode([p["abstract"]])[0]
                p_hex, _ = compute_iqd_hex(p_vec, manifest)
                baseline_hex_votes.append(p_hex)
                paper_records.append(f"- `[{p['id']}]` {p['title']}")
                for i in range(6):
                    if p_hex[i] == '1':
                        vote_stats[i] += 1
                
            baseline_hex = ""
            for i in range(6):
                baseline_hex += "1" if vote_stats[i] > (len(valid_papers) / 2) else "0"
        else:
            # 呈現破裂
            baseline_hex = "000000"
            paper_records.append(f"- (原始抓取 {raw_hits} 篇，但均無法與本體論產生語意共振)")
            paper_records.append("- (系統坦承觀測破裂，無法建立有效外部基準)")

        breakthrough_dims = [manifest["dimensions"][list(manifest["dimensions"].keys())[i]]["layer"] 
                             for i in range(6) if user_hex[i] == "1" and baseline_hex[i] == "0"]
        
        if "Valid" not in baseline_status:
            breakthrough_str = "基準破裂，無法進行相對維度測量"
        else:
            breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與傳統基準同頻"
        
        sys_prompt = f"""
你是一位專注於系統工程與物理本體論的客觀解讀者。
本理論經過實體文獻對比，測量狀態為：【{baseline_status}】。
理論狀態定位為：{user_state_info['name']}。
請根據下文撰寫約 200 字的「理論導讀摘要」。語氣客觀，禁止使用宣傳式修辭。
第一句話必須以「本理論架構...」開頭。
"""
        clean_summary = call_copilot_brain(sys_prompt, raw_text[:3000])

        return {
            "user_hex": user_hex,
            "baseline_hex": baseline_hex,
            "breakthrough_str": breakthrough_str,
            "state_name": user_state_info['name'],
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "top_sentence": top_sentence[:100] + "...",
                "arxiv_query": actual_query,
                "raw_hits": raw_hits,
                "valid_hits": len(valid_papers),
                "paper_records": paper_records,
                "vote_stats": vote_stats,
                "baseline_status": baseline_status,
                "embedding_model": EMBEDDING_MODEL_NAME,
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
        f"* **觀測儀器 (Vector)**：`{meta['embedding_model']}`\n\n"
        f"---\n"
        f"### 1. 🔬 觀測參數與基準建立 (Observation Parameters)\n"
        f"* **原著核心波包 (Peak Sentence)**：\n  > *\"{meta['top_sentence']}\"*\n"
        f"* **動態檢索條件 (Query)**：`{meta['arxiv_query']}`\n"
        f"* **基準校準狀態**：`{meta['baseline_status']}`\n"
        f"* **有效觀測樣本**：\n"
        f"{papers_text}\n\n"
        f"* **基準矩陣投票統計**：`[ {vote_str} ]`\n"
        f"* 🗺️ **外部真實基準 (Baseline Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"### 2. ⚖️ IQD 差分干涉結果 (Interference Results)\n"
        f"* 🛡️ **本體論絕對指紋**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* 🌌 **拓樸破缺維度 (相對於有效基準)**：**【{data['breakthrough_str']}】**\n\n"
        f"**詳細差分儀表板**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"> *註：本報告為系統自動化向量觀測紀錄，不包含生成式 AI 之主觀詮釋。*\n"
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
        "        <h3>📡 學術價值全像儀 (AVH) 實體觀測認證</h3>\n"
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
        log_file.write("# 📡 AVH 學術價值全像儀：純淨物理觀測日誌\n---\n")
        
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
