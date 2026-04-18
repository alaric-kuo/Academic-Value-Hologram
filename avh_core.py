import os
import sys
import json
import glob
import numpy as np
import requests
from openai import OpenAI
from datetime import datetime

# ==============================================================================
# AVH Genesis Engine (V2.1.0 完整軌跡收斂版)
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

def extract_ontological_trajectory(source_path):
    """提取全文連續積分、重心特徵與探針"""
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            full_text = " ".join(file.read().split())
    except Exception as e:
        print(f"檔案讀取異常，跳過此文件 ({str(e)})")
        return None
        
    if len(full_text) < 500:
        print(f"⚠️ {source_path} 文本資訊熵過低，忽略觀測。")
        return None
    
    window_size, stride = 1500, 800
    trajectories = [full_text[i:i+window_size] for i in range(0, len(full_text), stride) if len(full_text[i:i+window_size]) > 100]
    
    try:
        response = client.embeddings.create(input=trajectories, model="text-embedding-3-small", timeout=30)
        wave_functions = [np.array(data.embedding) for data in response.data]
        
        psi_global = np.mean(wave_functions, axis=0)
        
        vec_stats = {
            "dim": len(psi_global),
            "mean": float(np.mean(psi_global)),
            "std": float(np.std(psi_global)),
            "norm": float(np.linalg.norm(psi_global))
        }
        
        similarities = [np.dot(wf, psi_global) / (np.linalg.norm(wf) * np.linalg.norm(psi_global)) for wf in wave_functions]
        centroid_index = np.argmax(similarities)
        semantic_probe = trajectories[centroid_index][:200]
        
        return {
            "psi_global": psi_global,
            "vec_stats": vec_stats,
            "probe_text": semantic_probe,
            "window_count": len(trajectories),
            "centroid_sim": float(similarities[centroid_index]),
            "full_text": full_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 向量轉換過程超時 ({str(e)})")
        sys.exit(1)

def scan_background_field(query_text):
    """掃描背景網格，並回傳【所有】碰撞到的實體標題"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query_text[:120], "limit": 10, "fields": "citationCount,title"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        total_citations = sum(p.get('citationCount', 0) for p in data)
        collisions = [p.get('title') for p in data] # 抓取所有命中的標題
        return total_citations, collisions
    except requests.exceptions.RequestException as e:
        print(f"工具調用失敗，原因為 Semantic Scholar API 阻擋 ({str(e)})")
        sys.exit(1)

def generate_trajectory_log(target_file, trajectory_data, bg_energy, collisions, hex_code, manifest):
    """生成包含完整碰撞軌跡的 Log"""
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    # 動態組裝所有碰撞標題
    collisions_text = "\n".join([f"    {i+1}. {title}" for i, title in enumerate(collisions)]) if collisions else "    無碰撞紀錄"
    
    return f"""
## 📡 觀測軌跡：`{target_file}`
* **物理時間戳**：`{timestamp}`

### 1. 🌌 全文能勢集成 (Wave Function Integration)
* **解析窗格數**：`{trajectory_data['window_count']} 視窗 (1500/800 Overlap)`
* **1536維重心矩陣特徵**：
    * `均值 (Mean)`：{stats['mean']:.8f}
    * `標準差 (Std)`：{stats['std']:.8f}
    * `模長 (L2 Norm)`：{stats['norm']:.8f}

### 2. 🎯 語意重心提取 (Semantic Centroid Probe)
* **重心相似度**：`{trajectory_data['centroid_sim']:.4f}`
* **物理探針內容**：
    > "{trajectory_data['probe_text']}..."

### 3. 💥 場域碰撞分析 (Background Field Collision)
* **探針場域回聲 (Probe Echo)**：`{bg_energy}` (引用質量總和)
* **網格碰撞實體 (Full Collision Record)**：
{collisions_text}

### 4. 🧬 最終狀態坍縮 (Topological Collapse)
* **狀態陣列**：`[{hex_code}]`
* **物理相變**：**{hex_info['name']}**
* **學術指紋**：
    > {hex_info['desc']}

---
"""

def export_wordpress_html(basename, content, hex_code, state_name):
    html_template = f"""
<div class="avh-hologram-article">
    <div class="avh-content">
        {content.replace('\n', '<br>')}
    </div>
    <hr>
    <div class="avh-seal" style="border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;">
        <p><strong>📡 本理論已通過 學術價值全像儀 (AVH) 認證</strong></p>
        <p>當下演化狀態：[ {hex_code} ] - <strong>{state_name}</strong></p>
        <p>語意時間戳：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><em>本體論底層協議保護 | 瀚菱管理顧問 AJ Consulting</em></p>
    </div>
</div>
"""
    with open(f'WP_Ready_{basename}.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

def export_latex(basename, content, hex_code, state_name):
    tex_template = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{xeCJK}
\title{""" + basename + r"""}
\author{Alaric Kuo}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
本文章經由 AVH 學術價值全像儀觀測，當下演化狀態為 [""" + hex_code + r"""] """ + state_name + r"""。
\end{abstract}
""" + content.replace('#', '\section') + r"""
\end{document}
"""
    with open(f'{basename}_Archive.tex', 'w', encoding='utf-8') as f:
        f.write(tex_template)

if __name__ == "__main__":
    # 更新為 avh_manifest.json
    if not os.path.exists('avh_manifest.json'):
        print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
        sys.exit(1)
        
    with open('avh_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    source_files = []
    for ext in ["*.md", "*.tex"]:
        source_files.extend([f for f in glob.glob(ext) if f.lower() not in ['readme.md', 'avh_observation_log.md']])
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟動 AVH 引擎，共偵測到 {len(source_files)} 個波包等待坍縮...")
    
    # 更新為 AVH_OBSERVATION_LOG.md
    with open('AVH_OBSERVATION_LOG.md', 'w', encoding='utf-8') as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：多維觀測實作軌跡\n")
        log_file.write("*本文件詳實紀錄方法論實作過程中，知識波包從高維向量到三維投影的每一處相變，作為不可篡改之演化鐵證。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            trajectory_data = extract_ontological_trajectory(target_source)
            if not trajectory_data: continue
            
            bg_energy, collisions = scan_background_field(trajectory_data['probe_text'])
            
            ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
            hex_bits = ""
            psi = trajectory_data['psi_global']
            
            for key in ordered_dimensions:
                dim = manifest['dimensions'][key]
                v_pos = get_embedding(dim['pos_def'])
                v_neg = get_embedding(dim['neg_def'])
                
                sim_pos = np.dot(psi, v_pos) / (np.linalg.norm(psi) * np.linalg.norm(v_pos))
                sim_neg = np.dot(psi, v_neg) / (np.linalg.norm(psi) * np.linalg.norm(v_neg))
                hex_bits += "1" if sim_pos > sim_neg else "0"
            
            last_hex_code = hex_bits
            state_name = manifest['states'][hex_bits]['name']
            
            # 1. 寫入軌跡 Log
            report = generate_trajectory_log(target_source, trajectory_data, bg_energy, collisions, hex_bits, manifest)
            log_file.write(report)
            
            # 2. 輸出 WordPress HTML
            basename = os.path.splitext(target_source)[0]
            export_wordpress_html(basename, trajectory_data['full_text'], hex_bits, state_name)
            
            # 3. 輸出 LaTeX
            export_latex(basename, trajectory_data['full_text'], hex_bits, state_name)
            
            print(f"✅ {target_source} 理論收斂完成！ [{hex_bits}]")

    if last_hex_code:
        with open(os.environ.get('GITHUB_ENV', 'env.tmp'), 'a') as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
