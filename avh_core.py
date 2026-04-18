import os
import sys
import json
import glob
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
# ==============================================================================
# AVH Genesis Engine (V5.0.0 拓樸大腦：核心邏輯鍊萃取版)
# ==============================================================================
print("🧠 [載入核心] 正在啟動多語系神經網路模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
try:
    # 支援 50+ 語言的高維模型，能真正看懂繁體中文與複雜哲學邏輯
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"模型載入失敗：{str(e)}")
    sys.exit(1)
def extract_ontological_trajectory(source_path):
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        # 1. 概念碎片化：將文章以段落為基礎進行拆解，過濾掉過短的無意義斷句
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            print(f"⚠️ {source_path} 文本結構過於單一，資訊熵不足。")
            return None
            
        # 2. 神經投影：將所有段落打入高維空間
        print("🕸️ [邏輯建構] 正在建立語意拓樸網格...")
        embeddings = embedding_model.encode(paragraphs)
        
        # 3. 邏輯引力網格建構：計算段落間的 Cosine 相似度矩陣
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0) # 排除自我參照的無限迴圈
        
        # 4. 拓樸中心收斂 (Semantic PageRank)：找出邏輯連貫性最強的節點
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        # 5. 邏輯鏈排序與萃取
        # 依據「邏輯中心性」分數由高到低排序段落
        ranked_paragraphs = sorted(((scores[i], s, embeddings[i], i) for i, s in enumerate(paragraphs)), reverse=True)
        
        # 動態截斷：只取全文本最具核心邏輯影響力的前 35% 段落 (自動拋棄字典檔、附錄與離群雜訊)
        core_logic_size = max(3, int(len(paragraphs) * 0.35))
        core_chain_data = ranked_paragraphs[:core_logic_size]
        
        # 重新依據原文順序組合這些核心邏輯，形成「核心邏輯鏈摘要」
        core_chain_data_sorted = sorted(core_chain_data, key=lambda x: x[3])
        core_logic_text = "\n".join([item[1] for item in core_chain_data_sorted])
        
        # 取得純粹核心邏輯的波函數集合，計算全局指紋
        cohesive_wfs = [item[2] for item in core_chain_data]
        psi_global = np.mean(cohesive_wfs, axis=0)
        
        print(f"🛡️ [拓樸大腦] 已成功拋棄硬編碼與邊緣雜訊。從 {len(paragraphs)} 個段落中，精煉出 {core_logic_size} 個核心邏輯節點。")
        
        vec_stats = {
            "dim": len(psi_global),
            "mean": float(np.mean(psi_global)),
            "std": float(np.std(psi_global)),
            "norm": float(np.linalg.norm(psi_global))
        }
        
        # 物理探針：直接取出中心性分數最高的那「一」句話，這就是整篇文章的絕對奇點
        absolute_core_probe = ranked_paragraphs[0][1][:200]
        
        return {
            "psi_global": psi_global,
            "vec_stats": vec_stats,
            "probe_text": absolute_core_probe,
            "logic_chain_summary": core_logic_text,
            "window_count": core_logic_size,
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 拓樸萃取過程錯誤 ({str(e)})")
        sys.exit(1)

def generate_trajectory_log(target_file, trajectory_data, hex_code, manifest):
    """生成純粹的自我演化軌跡紀錄，附上 AI 自動摘要的核心邏輯"""
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    # 防止過長摘要擠爆版面，限制顯示前 800 字
    summary_display = trajectory_data['logic_chain_summary'][:800]
    if len(trajectory_data['logic_chain_summary']) > 800:
        summary_display += "...\n(後續邏輯鏈已截斷)"
    
    return f"""
## 📡 觀測軌跡：`{target_file}`
* **物理時間戳**：`{timestamp}`

### 1. 🧠 核心邏輯拓樸萃取 (Semantic Graph Abstraction)
* **邏輯節點數**：從全文提煉出 `{trajectory_data['window_count']}` 個具備最高引力的核心邏輯段落。
* **高維矩陣特徵** (排除字典與雜訊後之純粹本體)：
    * `均值 (Mean)`：{stats['mean']:.8f}
    * `標準差 (Std)`：{stats['std']:.8f}
    * `模長 (L2 Norm)`：{stats['norm']:.8f}

### 2. 🎯 理論奇點探針 (Absolute Logic Centroid)
* **全文最高維度交匯點 (核心主張)**：
    > "{trajectory_data['probe_text']}..."

### 3. 🧬 最終狀態坍縮 (Topological Collapse)
* **狀態陣列**：`[{hex_code}]`
* **物理相變**：**{hex_info['name']}**
* **學術指紋**：
    > {hex_info['desc']}

---
### 🔗 附錄：AI 拓樸大腦提煉之「核心邏輯鏈摘要」
*(系統自動判讀此波包的本體架構，已剔除無關之舉例、字典檔與離群雜訊)*
```text
{summary_display}



"""
def export_wordpress_html(basename, content, hex_code, state_name):
# 避開 Python 3.10 以下 f-string 無法使用反斜線的雷區
html_content = content.replace('\n', '
')
timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html_template = f"""

<div class="avh-hologram-article">
<div class="avh-content">
{html_content}
</div>
<hr>
<div class="avh-seal" style="border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;">
<p><strong>📡 本理論已通過 學術價值全像儀 (AVH) 核心邏輯檢驗</strong></p>
<p>當下演化狀態：[ {hex_code} ] - <strong>{state_name}</strong></p>
<p>語意時間戳：{timestamp_str}</p>
<p><em>拓樸大腦邏輯萃取 | 本體論底層協議保護 | AJ Consulting</em></p>
</div>
</div>
"""
with open(f'WP_Ready_{basename}.html', 'w', encoding='utf-8') as f:
f.write(html_template)
def export_latex(basename, content, hex_code, state_name):
tex_content = content.replace('#', '\section')
tex_template = (
"\documentclass{article}\n"
"\usepackage[utf8]{inputenc}\n"
"\usepackage{xeCJK}\n"
"\title{" + basename + "}\n"
"\author{Alaric Kuo}\n"
"\date{\today}\n"
"\begin{document}\n"
"\maketitle\n"
"\begin{abstract}\n"
"本文章經由 AVH 學術價值全像儀觀測，當下演化狀態為 [" + hex_code + "] " + state_name + "。\n"
"\end{abstract}\n\n"
+ tex_content + "\n\n"
"\end{document}\n"
)
with open(f'{basename}_Archive.tex', 'w', encoding='utf-8') as f:
f.write(tex_template)
if name == "main":
if not os.path.exists('avh_manifest.json'):
print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
sys.exit(1)
with open('avh_manifest.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)
    
source_files = []
# 掃描目標：所有的 md 與 tex 檔 (排除自動生成的 Log 避免無限迴圈)
for ext in ["*.md", "*.tex"]:
    source_files.extend([f for f in glob.glob(ext) if f.lower() not in ['avh_observation_log.md']])

if not source_files:
    print("系統休眠：未偵測到有效理論源碼波包。")
    sys.exit(0)
    
print(f"\n🚀 啟動 AVH 引擎 (拓樸大腦模式)，共偵測到 {len(source_files)} 個波包等待坍縮...")

with open('AVH_OBSERVATION_LOG.md', 'w', encoding='utf-8') as log_file:
    log_file.write("# 📡 AVH 學術價值全像儀：純粹邏輯定位軌跡\n")
    log_file.write("*本文件詳實紀錄方法論實作過程中，知識波包透過圖論 (Graph Theory) 萃取出絕對核心邏輯後，所坍縮出的最終拓樸狀態。*\n\n---\n")
    
    last_hex_code = ""
    for target_source in source_files:
        trajectory_data = extract_ontological_trajectory(target_source)
        if not trajectory_data: continue
        
        ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        hex_bits = ""
        psi = trajectory_data['psi_global']
        
        # 使用新大腦對六維定義進行動態編碼與碰撞
        for key in ordered_dimensions:
            dim = manifest['dimensions'][key]
            v_pos = embedding_model.encode([dim['pos_def']])[0]
            v_neg = embedding_model.encode([dim['neg_def']])[0]
            
            sim_pos = np.dot(psi, v_pos) / (np.linalg.norm(psi) * np.linalg.norm(v_pos))
            sim_neg = np.dot(psi, v_neg) / (np.linalg.norm(psi) * np.linalg.norm(v_neg))
            hex_bits += "1" if sim_pos > sim_neg else "0"
        
        last_hex_code = hex_bits
        state_name = manifest['states'][hex_bits]['name']
        
        # 寫入物理投影資產
        report = generate_trajectory_log(target_source, trajectory_data, hex_bits, manifest)
        log_file.write(report)
        
        basename = os.path.splitext(target_source)[0]
        export_wordpress_html(basename, trajectory_data['full_text'], hex_bits, state_name)
        export_latex(basename, trajectory_data['full_text'], hex_bits, state_name)
        
        print(f"✅ {target_source} 理論收斂完成！ [{hex_bits}]")

# 將狀態輸出給 CI/CD 進行 Tag 標註
if last_hex_code:
    with open(os.environ.get('GITHUB_ENV', 'env.tmp'), 'a') as env_file:
        env_file.write(f"HEX_CODE={last_hex_code}\n")
