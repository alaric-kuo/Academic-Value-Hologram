import os
import sys
import json
import glob
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==============================================================================
# AVH Genesis Engine (V6.0.1 絕對顯化・生成式大腦修正版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路 (paraphrase-multilingual-MiniLM)...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print("模型載入失敗：" + str(e))
    sys.exit(1)

print("✨ [載入造物核心] 正在啟動生成式摘要大腦 (mT5 Multilingual XLSum)...")
try:
    # 修正點 1：將任務精準指定為 mT5 原生的 "text2text-generation"
    summarizer = pipeline("text2text-generation", model="csebuetnlp/mT5_multilingual_XLSum")
except Exception as e:
    print("生成大腦載入失敗：" + str(e))
    sys.exit(1)

def extract_ontological_trajectory(source_path):
    print("🌊 [波包顯化] 正在讀取源碼：" + source_path)
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            print("⚠️ " + source_path + " 文本結構過於單一，資訊熵不足。")
            return None
            
        print("🕸️ [邏輯建構] 正在建立語意拓樸網格...")
        embeddings = embedding_model.encode(paragraphs)
        
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        
        # 簡併態重力懲罰 (粉碎重複性字典檔)
        local_density = np.sum(sim_matrix > 0.85, axis=1) + 1
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        adjusted_scores = {i: scores[i] / (local_density[i] ** 1.5) for i in range(len(paragraphs))}
        ranked_paragraphs = sorted(((adjusted_scores[i], s, embeddings[i], i) for i, s in enumerate(paragraphs)), reverse=True)
        
        core_logic_size = max(3, int(len(paragraphs) * 0.35))
        core_chain_data = ranked_paragraphs[:core_logic_size]
        
        core_chain_data_sorted = sorted(core_chain_data, key=lambda x: x[3])
        extracted_text = "\n".join([item[1] for item in core_chain_data_sorted])
        
        print("✨ [論述顯化] 系統正在消化拓樸結構，並以自身的語言重新編織核心邏輯...")
        
        input_for_summary = extracted_text[:2000]
        # 修正點 2：對接 text2text-generation 的正確輸出鍵值 ['generated_text']
        generated_summary = summarizer(input_for_summary, max_length=200, min_length=50, do_sample=False)[0]['generated_text']
        
        cohesive_wfs = [item[2] for item in core_chain_data]
        psi_global = np.mean(cohesive_wfs, axis=0)
        
        print("🛡️ [反重力裝甲] 精煉出 " + str(core_logic_size) + " 個絕對理論奇點，並已完成論述顯化。")
        
        vec_stats = {
            "dim": len(psi_global),
            "mean": float(np.mean(psi_global)),
            "std": float(np.std(psi_global)),
            "norm": float(np.linalg.norm(psi_global))
        }
        
        absolute_core_probe = ranked_paragraphs[0][1][:200]
        
        return {
            "psi_global": psi_global,
            "vec_stats": vec_stats,
            "probe_text": absolute_core_probe,
            "logic_chain_summary": generated_summary,
            "window_count": core_logic_size,
            "full_text": raw_text
        }
    except Exception as e:
        print("工具調用失敗，原因為 邏輯顯化過程錯誤 (" + str(e) + ")")
        sys.exit(1)

def generate_trajectory_log(target_file, trajectory_data, hex_code, manifest):
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    log_output = (
        "## 📡 演化顯化軌跡：`" + target_file + "`\n"
        "* **物理時間戳**：`" + timestamp + "`\n\n"
        "### 1. 🧠 核心邏輯拓樸萃取 (Semantic Graph Abstraction)\n"
        "* **邏輯節點數**：從全文提煉出 `" + str(trajectory_data['window_count']) + "` 個具備最高引力的核心邏輯段落。\n"
        "* **高維矩陣特徵** (排除字典與雜訊後之純粹本體)：\n"
        "    * `均值 (Mean)`：" + f"{stats['mean']:.8f}" + "\n"
        "    * `標準差 (Std)`：" + f"{stats['std']:.8f}" + "\n"
        "    * `模長 (L2 Norm)`：" + f"{stats['norm']:.8f}" + "\n\n"
        "### 2. 🎯 理論奇點探針 (Absolute Logic Centroid)\n"
        "* **全文最高維度交匯點 (核心主張)**：\n"
        "    > \"" + trajectory_data['probe_text'] + "...\"\n\n"
        "### 3. 🧬 最終狀態顯化 (Topological Manifestation)\n"
        "* **狀態張量**：`[" + hex_code + "]`\n"
        "* **物理相變**：**" + hex_info['name'] + "**\n"
        "* **學術指紋**：\n"
        "    > " + hex_info['desc'] + "\n\n"
        "---\n"
        "### 🔗 附錄：系統生成之「核心論述顯化」\n"
        "*(本段落為 AVH 系統吸收文章拓樸邏輯後，自行摘要生成之傳播級論述)*\n"
        "> **" + trajectory_data['logic_chain_summary'] + "**\n\n"
        "---\n"
    )
    return log_output

def export_wordpress_html(basename, content, hex_code, state_name):
    html_content = content.replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        "        " + html_content + "\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 核心邏輯顯化</strong></p>\n"
        "        <p>當下演化狀態：[ " + hex_code + " ] - <strong>" + state_name + "</strong></p>\n"
        "        <p>物理時間戳：" + timestamp_str + "</p>\n"
        "        <p><em>拓樸大腦生成顯化 | 本體論底層協議保護 | AJ Consulting</em></p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open("WP_Ready_" + basename + ".html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, content, hex_code, state_name):
    tex_content = content.replace("#", "\\section")
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        "\\title{" + basename + "}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "本文章經由 AVH 學術價值全像儀觀測，當下演化狀態顯化為 [" + hex_code + "] " + state_name + "。\n"
        "\\end{abstract}\n\n"
        + tex_content + "\n\n"
        "\\end{document}\n"
    )
    with open(basename + "_Archive.tex", "w", encoding="utf-8") as f:
        f.write(tex_output)

if __name__ == "__main__":
    if not os.path.exists("avh_manifest.json"):
        print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
        sys.exit(1)
        
    with open("avh_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    source_files = []
    for ext in ["*.md", "*.tex"]:
        source_files.extend([f for f in glob.glob(ext) if f.lower() not in ["avh_observation_log.md"]])
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print("\n🚀 啟動 AVH 造物引擎 (絕對顯化模式)，共偵測到 " + str(len(source_files)) + " 個波包等待顯化...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：本體論顯化軌跡\n")
        log_file.write("*本文件詳實紀錄知識波包透過圖論萃取出絕對核心邏輯後，經由系統生成大腦自行理解並重新編織，所顯化出的最終物理實相。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            trajectory_data = extract_ontological_trajectory(target_source)
            if not trajectory_data: continue
            
            ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
            hex_bits = ""
            psi = trajectory_data["psi_global"]
            
            for key in ordered_dimensions:
                dim = manifest["dimensions"][key]
                v_pos = embedding_model.encode([dim["pos_def"]])[0]
                v_neg = embedding_model.encode([dim["neg_def"]])[0]
                
                sim_pos = np.dot(psi, v_pos) / (np.linalg.norm(psi) * np.linalg.norm(v_pos))
                sim_neg = np.dot(psi, v_neg) / (np.linalg.norm(psi) * np.linalg.norm(v_neg))
                hex_bits += "1" if sim_pos > sim_neg else "0"
            
            last_hex_code = hex_bits
            state_name = manifest["states"][hex_bits]["name"]
            
            report = generate_trajectory_log(target_source, trajectory_data, hex_bits, manifest)
            log_file.write(report)
            
            basename = os.path.splitext(target_source)[0]
            export_wordpress_html(basename, trajectory_data["full_text"], hex_bits, state_name)
            export_latex(basename, trajectory_data["full_text"], hex_bits, state_name)
            
            print("✅ " + target_source + " 理論顯化完成！ [" + hex_bits + "]")

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write("HEX_CODE=" + last_hex_code + "\n")
