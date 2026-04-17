import os
import sys
import json
import glob
import numpy as np
import requests
from openai import OpenAI
from datetime import datetime

# ==============================================================================
# QTE Academic Hologram Core Engine (V1.2.0 批次觀測版)
# ==============================================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
S2_API_KEY = os.environ.get("S2_API_KEY", "")

def get_embedding(text):
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small", timeout=15)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"工具調用失敗，原因為 OpenAI API 拒絕連線或超時狀態 ({str(e)})")
        sys.exit(1)

def extract_continuous_fingerprint(source_path):
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            full_text = " ".join(file.read().split())
    except Exception as e:
        print(f"檔案讀取異常，跳過此文件 ({str(e)})")
        return None, None
        
    if len(full_text) < 500:
        print(f"⚠️ {source_path} 文本資訊熵過低，忽略觀測。")
        return None, None
    
    # 修正：精準對齊白皮書宣告的視窗參數
    window_size, stride = 1500, 800
    trajectories = [full_text[i:i+window_size] for i in range(0, len(full_text), stride) if len(full_text[i:i+window_size]) > 100]
    
    try:
        response = client.embeddings.create(input=trajectories, model="text-embedding-3-small", timeout=30)
        wave_functions = [np.array(data.embedding) for data in response.data]
        return full_text[:200], np.mean(wave_functions, axis=0)
    except Exception as e:
        print(f"工具調用失敗，原因為 向量轉換過程超時 ({str(e)})")
        sys.exit(1)

def scan_background_field(query_text):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query_text[:120], "limit": 10, "fields": "citationCount"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"工具調用失敗，原因為 Semantic Scholar API 阻擋 ({str(e)})")
        sys.exit(1)
        
    return sum(p.get('citationCount', 0) for p in response.json().get('data', []))

def validate_semantic_coherence(psi_global):
    baseline_text = "Academic research paper, theoretical framework, engineering application, scientific methodology, ontology, topology."
    baseline_vector = get_embedding(baseline_text)
    sim = np.dot(psi_global, baseline_vector) / (np.linalg.norm(psi_global) * np.linalg.norm(baseline_vector))
    if sim < 0.15: 
        return False
    return True

def calculate_hexagram(psi_global, manifest):
    # 修正：強制綁定陣列順序，徹底解除 JSON Key 的順序依賴
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    hex_bits = ""
    
    for key in ordered_dimensions:
        dim = manifest['dimensions'][key]
        v_pos = get_embedding(dim['pos_def'])
        v_neg = get_embedding(dim['neg_def'])
        
        sim_pos = np.dot(psi_global, v_pos) / (np.linalg.norm(psi_global) * np.linalg.norm(v_pos))
        sim_neg = np.dot(psi_global, v_neg) / (np.linalg.norm(psi_global) * np.linalg.norm(v_neg))
        hex_bits += "1" if sim_pos > sim_neg else "0"
        
    return hex_bits

def generate_hologram_report(target_file, hex_code, energy, manifest):
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "系統偵測到未知演化路徑，資訊熵溢出。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    
    return f"""
### 📄 觀測目標：`{target_file}`
* **語意時間戳**：`{timestamp}`

#### 🌌 本體論六爻投影 
* **狀態陣列**：`[{hex_code}]`
* **物理相變**：**{hex_info['name']}**
* **初探波包能勢**：`{energy}` *(註：此為基於前端探針的學術場域引用質量，非全文絕對能量)*

#### 🧬 QTE 演化軌跡判讀
> **{hex_info['desc']}**

#### 📐 拓樸維度解析 (天地人)
* **[天] 價值意圖與治理維度:** `[{hex_code[0]}, {hex_code[1]}]`
* **[人] 認知深度與描述架構:** `[{hex_code[2]}, {hex_code[3]}]`
* **[地] 擴張潛力與應用實相:** `[{hex_code[4]}, {hex_code[5]}]`

---
"""

if __name__ == "__main__":
    if not os.path.exists('qte_academic_manifest.json'):
        print("工具調用失敗，原因為 遺失底層定義檔")
        sys.exit(1)
        
    with open('qte_academic_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    # 修正：批次讀取，且嚴格排除 README.md 與輸出報告檔
    md_files = [f for f in glob.glob("*.md") if f.lower() not in ['readme.md', 'qte_observation_log.md']]
    
    if not md_files:
        print("系統休眠：未偵測到有效 Markdown 源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟動 QTE 引擎，共偵測到 {len(md_files)} 個波包等待坍縮...")
    
    # 準備寫入獨立的觀測日誌
    with open('QTE_OBSERVATION_LOG.md', 'w', encoding='utf-8') as log_file:
        log_file.write("# 📡 QTE 學術全像儀：多維觀測日誌\n")
        log_file.write("*本報告由 QTE 底層協議自動生成，詳實紀錄各知識波包的拓樸狀態。*\n\n---\n")
        
        last_hex_code = ""
        
        for target_source in md_files:
            probe_text, psi = extract_continuous_fingerprint(target_source)
            if psi is None: continue
            
            if not validate_semantic_coherence(psi):
                print(f"⚠️ {target_source} 邏輯崩潰，拒絕投影。")
                continue
                
            bg_energy = scan_background_field(probe_text)
            hex_code = calculate_hexagram(psi, manifest)
            last_hex_code = hex_code # 供 Github Actions 抓取最後一個處理的狀態
            
            report = generate_hologram_report(target_source, hex_code, bg_energy, manifest)
            log_file.write(report)
            print(f"✅ {target_source} 物理投影完成！ [{hex_code}]")

    # 輸出環境變數給 CI/CD
    if last_hex_code:
        with open(os.environ.get('GITHUB_ENV', 'env.tmp'), 'a') as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
"""
