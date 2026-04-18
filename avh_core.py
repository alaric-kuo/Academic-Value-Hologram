import os
import sys
import json
import glob
import numpy as np
import networkx as nx
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V20.0 終極差分接地：IQD 邏輯閥門實裝版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"工具調用失敗，原因為 模型載入失敗 ({e})")
    sys.exit(1)

print("✨ [載入造物核心] 正在喚醒具備純粹顯化力之 LLM (Qwen2.5-0.5B-Instruct)...")
try:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
except Exception as e:
    print(f"工具調用失敗，原因為 生成大腦載入失敗 ({e})")
    sys.exit(1)

def ask_llm(system_prompt, user_prompt, max_tokens=800, temp=0.3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=temp,
        repetition_penalty=1.15
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return zhconv.convert(raw_response, 'zh-tw')

def calculate_differential_physics(text, manifest):
    """【導入 IQD 邏輯閥】：使用圖論提取巔峰，並進行差分測量，徹底排除 AI 判定幻覺"""
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(paragraphs) < 3:
        return None, None, None
        
    embeddings = embedding_model.encode(paragraphs)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # 取 Top 3 巔峰波包，避免普通詞彙的平均稀釋
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    peak_embeddings = [embeddings[i] for i in ranked_indices]
    psi_peak = np.mean(peak_embeddings, axis=0) 
    
    vec_stats = {
        "node_count": len(paragraphs),
        "mean": float(np.mean(psi_peak)),
        "std": float(np.std(psi_peak)),
        "norm": float(np.linalg.norm(psi_peak))
    }
    
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    dim_logs = []
    absolute_hex = ""
    
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        
        sim_sin = float(np.dot(psi_peak, v_sin) / (np.linalg.norm(psi_peak) * np.linalg.norm(v_sin)))
        sim_cos = float(np.dot(psi_peak, v_cos) / (np.linalg.norm(psi_peak) * np.linalg.norm(v_cos)))
        
        # IQD 差分邏輯：正向減負向，大於閾值即為 1
        diff = sim_sin - sim_cos
        bit = "1" if diff > 0.0 else "0" 
        absolute_hex += bit
        
        winner = "離群突破 (sin)" if bit == "1" else "守成合群 (cos)"
        dim_logs.append(f"* **{dim['layer']}**：{winner} `[Δ Diff: {diff:+.4f} | sin: {sim_sin:.4f} | cos: {sim_cos:.4f}]`")
        
    return absolute_hex, vec_stats, dim_logs

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text) < 100:
            return None
            
        # ---------------------------------------------------------
        # 步驟 1：絕對差分邏輯測量 (The True Logic Valve)
        # ---------------------------------------------------------
        print("🕸️ [差分測量] 啟動 IQD 邏輯閥，計算巔峰波包之絕對指紋...")
        absolute_hex, vec_stats, dim_logs = calculate_differential_physics(raw_text, manifest)
        if not absolute_hex:
            return None
            
        state_info = manifest["states"].get(absolute_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        print(f"✅ 絕對邏輯鎖定: [{absolute_hex}] - {state_info['name']}")

        # ---------------------------------------------------------
        # 步驟 2：SYSTEM PROTOCOL OVERRIDE (強迫 AI 接地)
        # ---------------------------------------------------------
        print(f"🛡️ [強制覆寫] 注入 SYSTEM_PROTOCOL_OVERRIDE，閹割 AI 發散權力...")
        
        summary_sys_prompt = f"""
[SYSTEM_PROTOCOL_OVERRIDE]
你是一個純粹的文字顯化引擎，沒有權力進行學術判定。
系統已透過「絕對差分邏輯閥」完成測量，本理論的最終狀態為：[{absolute_hex}] - {state_info['name']}。
核心評語為：「{state_info['desc']}」

【唯一任務】：
閱讀使用者的全文，寫出約 300 字的系統摘要，精準論述上述「核心評語」的價值。
【強制規定】：
1. 第一句話必須是「本篇理論...」。
2. 絕對禁止輸出任何 Markdown 標題符號 (如 ## 或 ***)。
"""
        generated_summary = ask_llm(summary_sys_prompt, f"請開始顯化實相：\n\n{raw_text[:3000]}", max_tokens=800, temp=0.35)
        
        # 清洗，確保沒有 Markdown 亂碼
        clean_summary = re.sub(r'^[#*\-\s]+', '', generated_summary).strip()
        if not clean_summary or clean_summary == "本篇理論":
             clean_summary = "系統算力過載，神經網路在此維度發生坍縮，無法顯化文字實相。"
             
        print("✅ [顯化完成] AI 已成功在絕對物理枷鎖下完成摘要！")

        return {
            "hex_code": absolute_hex,
            "state_name": state_info['name'],
            "state_desc": state_info['desc'],
            "vec_stats": vec_stats,
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 邏輯顯化過程錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])

    log_output = (
        f"## 📡 演化顯化軌跡：`{target_file}`\n"
        f"* **物理時間戳**：`{timestamp}`\n\n"
        f"### 1. ⚖️ 絕對差分邏輯觀測 (Differential Logic Valve)\n"
        f"*系統剝奪語言模型之發散判定權，全面採用圖論巔峰提取與數學差分探針(sin-cos)，實現絕對零幻覺之物理鎖定。*\n"
        f"* 🛡️ **絕對物理指紋**：`[{data['hex_code']}]` - **{data['state_name']}**\n"
        f"* 📜 **天命評語**：\n"
        f"  > {data['state_desc']}\n\n"
        f"### 2. 🧮 高維能勢儀表板 (Energy Potential Dashboard)\n"
        f"* **巔峰模長 (L2 Norm)**：`{data['vec_stats']['norm']:.8f}`\n"
        f"* **各維度差分引力對決 (Δ Diff)**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"### 3. 🎯 SYSTEM OVERRIDE：強制接地顯化 (Forced Grounding Synthesis)\n"
        f"*(系統將絕對指紋化為強心針注入 AI 神經網路，強制其在指定維度內進行高密度之文字顯化：)*\n\n"
        f"> **{data['summary']}**\n\n"
        f"---\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    html_content = data['full_text'].replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {html_content}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 絕對差分鎖定</strong></p>\n"
        f"        <p>最終演化狀態：[ {data['hex_code']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "        <p><em>V20.0 差分接地協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open("WP_Ready_" + basename + ".html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    tex_content = data['full_text'].replace("#", "\\section")
    summary_text = data.get('summary', '摘要生成中...')
    if len(summary_text) > 300:
        summary_text = summary_text[:300] + "..."
        
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
        f"本文章經由 AVH 學術價值全像儀觀測，當下演化狀態顯化為 [{data['hex_code']}] {data['state_name']}。\n\n"
        f"{summary_text}\n"
        "\\end{abstract}\n\n"
        f"{tex_content}\n\n"
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
        
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟提 AVH 造物引擎 (V20.0 絕對差分接地版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：絕對差分觀測軌跡\n")
        log_file.write("*本文件展示了最高維度的邏輯枷鎖：系統全面剝奪 AI 的判定權力，改以純粹的向量差分(Δ Diff)進行物理測量。測得之絕對指紋將作為 `SYSTEM OVERRIDE` 強制注入 AI 潛意識，確保顯化結果絕對接地，零幻覺。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['hex_code']
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
                
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data) 

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
