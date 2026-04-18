import os
import sys
import json
import glob
import numpy as np
import requests
from datetime import datetime

# 導入開源本地端運算模型 (免 API Key)
from sentence_transformers import SentenceTransformer

# ==============================================================================
# AVH Genesis Engine (V3.0.0 零金鑰・算力解放版)
# ==============================================================================

# 初始化本地開源模型 (384維度，極度輕量且適合語意比對)
print("🧠 [載入核心] 正在啟動開源神經網路模型 (all-MiniLM-L6-v2)...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"模型載入失敗：{str(e)}")
    sys.exit(1)

# Semantic Scholar 免費流 (無 Key 亦可運行)
S2_API_KEY = os.environ.get("S2_API_KEY", "")

def get_embedding(text):
    """使用本地模型計算向量"""
    return embedding_model.encode([text])[0]

def extract_ontological_trajectory(source_path):
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            full_text = " ".join(file.read().split())
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
    except Exception as e:
        print(f"檔案讀取異常，跳過此文件 ({str(e)})")
        return None
        
    if len(full_text) < 500:
        print(f"⚠️ {source_path} 文本資訊熵過低，忽略觀測。")
        return None
    
    window_size, stride = 1500, 800
    trajectories = [full_text[i:i+window_size] for i in range(0, len(full_text), stride) if len(full_text[i:i+window_size]) > 100]
    
    # 讓 GitHub 的 CPU 免費幫我們計算
    wave_functions = [embedding_model.encode([chunk])[0] for chunk in trajectories]
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
        "full_text": raw_text
    }

def scan_background_field(query_text):
    """掃描背景網格 (免費流：無 Key 狀態下依然運作)"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query_text[:120], "limit": 10, "fields": "citationCount,title"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        total_citations = sum(p.get('citationCount', 0) for p in data)
        collisions = [p.get('title') for p in data]
        return total_citations, collisions
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Semantic Scholar API 阻擋或超時 (免費流限制)，本次略過場域回聲 ({str(e)})")
        return 0, []

def generate_trajectory_log(target_file, trajectory_data, bg_energy, collisions, hex_code, manifest):
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    collisions_list = []
    for i, title in enumerate(collisions):
        collisions_list.append(f"    {i+1}. {title}")
    collisions_text = "\n".join(collisions_list) if collisions_list else "    無碰撞紀錄 (新突破點)"
    
    return f"""
## 📡 觀測軌跡：`{target_file}`
* **物理時間戳**：`{timestamp}`

### 1. 🌌 全文能勢集成 (Wave Function Integration)
* **解析窗格數**：`{trajectory_data['window_count']} 視窗 (1500/800 Overlap)`
* **384維重心矩陣特徵** (無伺服器開源運算)：
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
    html_content = content.replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_template = f"""
<div class="avh-hologram-article">
    <div class="avh-content">
        {html_content}
    </div>
    <hr>
    <div class="avh-seal" style="border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;">
        <p><strong>📡 本理論已通過 學術價值全像儀 (AVH) 認證</strong></p>
        <p>當下演化狀態：[ {hex_code} ] - <strong>{state_name}</strong></p>
        <p>語意時間戳：{timestamp_str}</p>
        <p><em>本體論底層協議保護 | 零金鑰開源實踐 | AJ Consulting</em></p>
    </div>
</div>
"""
    with open(f'WP_Ready_{basename}.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

def export_latex(basename, content, hex_code, state_name):
    tex_content = content.replace('#', '\\section')
    tex_template = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        "\\title{" + basename + "}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "本文章經由 AVH 學術價值全像儀觀測，當下演化狀態為 [" + hex_code + "] " + state_name + "。\n"
        "\\end{abstract}\n\n"
        + tex_content + "\n\n"
        "\\end{document}\n"
    )
    with open(f'{basename}_Archive.tex', 'w', encoding='utf-8') as f:
        f.write(tex_template)

if __name__ == "__main__":
    if not os.path.exists('avh_manifest.json'):
        print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
        sys.exit(1)
        
    with open('avh_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    source_files = []
    for ext in ["*.md", "*.tex"]:
        # ⚠️ 已經拿掉 'readme.md' 的排除，將其視為最高優先級的知識波包
        source_files.extend([f for f in glob.glob(ext) if f.lower() not in ['avh_observation_log.md']])
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟動 AVH 引擎 (零金鑰模式)，共偵測到 {len(source_files)} 個波包等待坍縮...")
    
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
            
            report = generate_trajectory_log(target_source, trajectory_data, bg_energy, collisions, hex_bits, manifest)
            log_file.write(report)
            
            basename = os.path.splitext(target_source)[0]
            export_wordpress_html(basename, trajectory_data['full_text'], hex_bits, state_name)
            export_latex(basename, trajectory_data['full_text'], hex_bits, state_name)
            
            print(f"✅ {target_source} 理論收斂完成！ [{hex_bits}]")

    if last_hex_code:
        with open(os.environ.get('GITHUB_ENV', 'env.tmp'), 'a') as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
