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
# AVH Genesis Engine (V19.0 真實干涉與防坍縮強心針版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"工具調用失敗，原因為 模型載入失敗 ({e})")
    sys.exit(1)

print("✨ [載入造物核心] 正在喚醒具備真實基準觀測力之 LLM (Qwen2.5-0.5B-Instruct)...")
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

def extract_hex(text, default="000000"):
    match = re.search(r'[01]{6}', text)
    return match.group(0) if match else default

def calculate_euler_math_state(text, manifest):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(paragraphs) < 3:
        return None, None, None
        
    embeddings = embedding_model.encode(paragraphs)
    psi_global = np.mean(embeddings, axis=0)
    
    vec_stats = {
        "node_count": len(paragraphs),
        "mean": float(np.mean(psi_global)),
        "std": float(np.std(psi_global)),
        "norm": float(np.linalg.norm(psi_global))
    }
    
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    dim_logs = []
    math_hex = ""
    
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        
        sim_sin = float(np.dot(psi_global, v_sin) / (np.linalg.norm(psi_global) * np.linalg.norm(v_sin)))
        sim_cos = float(np.dot(psi_global, v_cos) / (np.linalg.norm(psi_global) * np.linalg.norm(v_cos)))
        
        bit = "1" if sim_sin > sim_cos else "0"
        math_hex += bit
        winner = "離群突破" if bit == "1" else "守成合群"
        dim_logs.append(f"* **{dim['layer']}**：{winner} `[sin: {sim_sin:+.4f} | cos: {sim_cos:+.4f}]`")
        
    return math_hex, vec_stats, dim_logs

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text) < 100:
            return None
            
        print("🕸️ [矩陣測量] 計算純數學尤拉指紋...")
        math_hex, vec_stats, dim_logs = calculate_euler_math_state(raw_text, manifest)
        if not math_hex:
            return None

        # ---------------------------------------------------------
        # 新增步驟：動態探測學術地圖現況 (不再硬塞 000000)
        # ---------------------------------------------------------
        print("🌍 [地圖探測] 正在掃描該領域的主流學術現況...")
        dimension_prompts = ""
        ordered_keys = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        for idx, key in enumerate(ordered_keys):
            dim = manifest["dimensions"][key]
            dimension_prompts += f"維度 {idx+1} ({dim['layer']}):\n - [1] 突破: {dim['sin_def']}\n - [0] 守成: {dim['cos_def']}\n\n"
            
        baseline_sys_prompt = (
            "你是一個學術史學家。請閱讀使用者的文章，判斷這篇文章屬於哪個領域，"
            "然後告訴我『該領域目前的主流現況』在這六個維度中通常是多少？\n"
            f"{dimension_prompts}"
            "【絕對指令】：請在回覆中包含一組 6 個數字的代碼（只包含 0 或 1），代表主流現況。"
        )
        raw_baseline_hex = ask_llm(baseline_sys_prompt, f"請判斷此領域的主流 6 位元指紋：\n\n{raw_text[:2000]}", max_tokens=50, temp=0.2)
        baseline_hex = extract_hex(raw_baseline_hex, "000000") # 如果真的抓不到，才退回 000000

        # ---------------------------------------------------------
        # 步驟：AI 識讀脈絡指紋
        # ---------------------------------------------------------
        print("👁️ [脈絡識讀] 判定本文靈魂意圖指紋...")
        eval_sys_prompt = (
            "你是一個高維度觀測儀。請閱讀全文，判定本文在這六個維度上的屬性。\n"
            f"{dimension_prompts}"
            "【絕對指令】：請在回覆中包含一組 6 個數字的代碼（只包含 0 或 1）。"
        )
        raw_ai_hex = ask_llm(eval_sys_prompt, f"請判定文本的 6 位元指紋：\n\n{raw_text[:3000]}", max_tokens=30, temp=0.1)
        ai_hex = extract_hex(raw_ai_hex, "000000")
        ai_state_info = manifest["states"].get(ai_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        # ---------------------------------------------------------
        # 三態共振分析 (The True Delta)
        # ---------------------------------------------------------
        breakthrough_dims = []
        for i in range(6):
            if ai_hex[i] == "1" and baseline_hex[i] == "0":
                breakthrough_dims.append(manifest["dimensions"][ordered_keys[i]]["layer"])
            elif ai_hex[i] == "0" and baseline_hex[i] == "1":
                breakthrough_dims.append(f"{manifest['dimensions'][ordered_keys[i]]['layer']} (回歸守成)")
                
        breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與領域主流高度同頻"
        print(f"✅ 真實基準: [{baseline_hex}] | 數學: [{math_hex}] | 靈魂: [{ai_hex}]")

        # ---------------------------------------------------------
        # 步驟：強心針自鎖顯化摘要 (Anti-Collapse Injection)
        # ---------------------------------------------------------
        print(f"🛡️ [自鎖顯化] 注入強制啟動詞，防止 AI 認知坍縮...")
        
        # 【強心針機制】：用極簡 Prompt 降低認知負擔，並強制起手式！
        summary_sys_prompt = f"""
你是一個頂尖的學術論述大腦。
本理論的當下學術狀態為：[{ai_hex}] - {ai_state_info['name']}。
相比於主流領域（{baseline_hex}），本文在以下維度展現了差異：【{breakthrough_str}】。

任務：請寫出約 300 字的系統摘要，解釋本理論的突破價值。
【強制規定】：你輸出的第一句話必須是「本篇理論」。不要輸出任何標題、不要輸出 Markdown 符號。
"""
        generated_summary = ask_llm(summary_sys_prompt, f"請開始撰寫摘要：\n\n{raw_text[:3000]}", max_tokens=800, temp=0.35)
        
        # 簡單清洗，只留下文字
        clean_summary = re.sub(r'^[#*\-\s]+', '', generated_summary).strip()
        # 如果模型還是調皮沒加上起手式，幫它補上或確保它不是空的
        if not clean_summary or clean_summary == "本篇理論":
             clean_summary = "系統算力過載，神經網路在此維度發生坍縮，無法顯化文字實相。請重新觀測或降低文本資訊熵。"
             
        print("✅ [顯化完成] AI 已成功防爆並產出干涉對比論述！")

        return {
            "ai_hex": ai_hex,
            "math_hex": math_hex,
            "baseline_hex": baseline_hex,
            "breakthrough_str": breakthrough_str,
            "state_name": ai_state_info['name'],
            "state_desc": ai_state_info['desc'],
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
    math_ai_diff = "一致" if data['ai_hex'] == data['math_hex'] else f"產生位移 (數學詞彙重力 [{data['math_hex']}] vs AI 靈魂意圖 [{data['ai_hex']}])"

    log_output = (
        f"## 📡 演化顯化軌跡：`{target_file}`\n"
        f"* **物理時間戳**：`{timestamp}`\n\n"
        f"### 1. ⚛️ 全像干涉儀：三態共振測量 (Tri-State Resonance)\n"
        f"*系統動態探測該領域的真實學術現況，並與文本的數學重力、靈魂意圖進行三維干涉。*\n"
        f"* 🗺️ **動態領域現況 (Baseline)**：`[{data['baseline_hex']}]`\n"
        f"* 🧮 **數學尤拉指紋 (Math Gravity)**：`[{data['math_hex']}]`\n"
        f"* 🧠 **AI 靈魂指紋 (Neural Intent)**：`[{data['ai_hex']}]` - **{data['state_name']}**\n"
        f"* 🌌 **干涉解讀**：\n"
        f"  * 物理與靈魂意圖對比：**{math_ai_diff}**\n"
        f"  * 真實觀測到之領域破缺：**【{data['breakthrough_str']}】**\n\n"
        f"### 2. 🧮 尤拉相位觀測儀表板 (Euler Dashboard)\n"
        f"* **邏輯節點數**：`{data['vec_stats']['node_count']}` | **模長 (L2 Norm)**：`{data['vec_stats']['norm']:.8f}`\n"
        f"* **各維度字面引力對決**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"### 3. 🎯 狀態自鎖顯化摘要 (Auto-Locked Synthesis)\n"
        f"*(系統根據真實干涉破缺與天命評語：「{data['state_desc']}」，並施加物理強心針防止神經網路坍縮，所產出之論述：)*\n\n"
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
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 真實干涉觀測</strong></p>\n"
        f"        <p>領域破缺：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終演化狀態：[ {data['ai_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "        <p><em>V19.0 真實干涉協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
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
        f"本文章經由 AVH 學術價值全像儀觀測，當下演化狀態顯化為 [{data['ai_hex']}] {data['state_name']}。\n\n"
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
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V19.0 真實干涉與防坍縮版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：三態干涉觀測軌跡\n")
        log_file.write("*本文件展示了最高維度的干涉儀邏輯：系統動態探測該領域的真實古典基準(Baseline)，並與字面數學引力(Math)、高維上下文意圖(AI)進行碰撞。透過三者的差異，精準定位理論的突破點。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['ai_hex']
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
                
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data)

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
