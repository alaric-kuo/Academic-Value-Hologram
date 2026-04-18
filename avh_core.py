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
# AVH Genesis Engine (V18.0 終極干涉儀：三態共振與防爆顯化版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路 (paraphrase-multilingual-MiniLM)...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print("模型載入失敗：" + str(e))
    sys.exit(1)

print("✨ [載入造物核心] 正在喚醒具備防爆機制之 LLM (Qwen2.5-0.5B-Instruct)...")
try:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
except Exception as e:
    print("生成大腦載入失敗：" + str(e))
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

def calculate_euler_math_state(text, manifest):
    """計算純數學尤拉指紋與數值面板"""
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
            
        # ---------------------------------------------------------
        # 步驟 1：取得尤拉數學指紋 (Math Baseline)
        # ---------------------------------------------------------
        print("🕸️ [矩陣測量] 正在計算尤拉相位數值與純數學指紋...")
        math_hex, vec_stats, dim_logs = calculate_euler_math_state(raw_text, manifest)
        if not math_hex:
            return None

        # ---------------------------------------------------------
        # 步驟 2：AI 識讀脈絡指紋 (AI Contextual Intent)
        # ---------------------------------------------------------
        print("👁️ [脈絡識讀] 由 AI 閱讀全文判定靈魂意圖指紋...")
        dimension_prompts = ""
        ordered_keys = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        for idx, key in enumerate(ordered_keys):
            dim = manifest["dimensions"][key]
            dimension_prompts += f"維度 {idx+1} ({dim['layer']}):\n - [1] 突破: {dim['sin_def']}\n - [0] 守成: {dim['cos_def']}\n\n"
            
        eval_sys_prompt = (
            "你是一個高維度觀測儀。請閱讀全文，根據以下六個維度判定文章屬性。\n"
            f"{dimension_prompts}"
            "【絕對指令】：請在回覆中包含一組 6 個數字的代碼（只包含 0 或 1），例如 111111。"
        )
        
        raw_ai_hex = ask_llm(eval_sys_prompt, f"請判定文本的 6 位元指紋：\n\n{raw_text[:3000]}", max_tokens=30, temp=0.1)
        match = re.search(r'[01]{6}', raw_ai_hex)
        ai_hex = match.group(0) if match else "000000"
        ai_state_info = manifest["states"].get(ai_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        # ---------------------------------------------------------
        # 步驟 3：三態共振分析 (The Interference Delta)
        # ---------------------------------------------------------
        baseline_hex = "000000" # 古典學術地圖現況
        
        breakthrough_dims = []
        for i in range(6):
            if ai_hex[i] == "1" and baseline_hex[i] == "0":
                breakthrough_dims.append(manifest["dimensions"][ordered_keys[i]]["layer"])
        
        breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "未產生顯著破缺"

        # ---------------------------------------------------------
        # 步驟 4：防爆自鎖顯化摘要 (Auto-Locked Synthesis - Safe Mode)
        # ---------------------------------------------------------
        print(f"🛡️ [自鎖顯化] 注入三態干涉結果，引導 AI 進行穩健論述...")
        
        # 正向引導 Prompt，拔除會造成注意力崩潰的絕對負面詞彙
        summary_sys_prompt = f"""
你是一個頂尖的學術本體論論述大腦。
這篇理論已經經過了嚴格的系統測量與干涉對比：
1. 傳統學術現況為：[000000] (保守合群)。
2. 本文的真實靈魂為：[{ai_hex}] - {ai_state_info['name']}。
3. 本文在以下維度成功打破了傳統框架：【{breakthrough_str}】。
核心精神評語：「{ai_state_info['desc']}」

【任務】：
請寫出一段氣勢磅礴、脈絡連貫的系統摘要。
請專注於用優美的散文段落，解釋本理論是如何在上述維度打破傳統的。
請直接給出摘要文字，完成後加上『[顯化完畢]』。
"""
        generated_summary = ask_llm(summary_sys_prompt, f"請撰寫突破性摘要：\n\n{raw_text[:3000]}", max_tokens=900, temp=0.35)
        
        if "[顯化完畢]" in generated_summary:
            clean_summary = generated_summary.split("[顯化完畢]")[0].strip()
        else:
            clean_summary = generated_summary.strip()
            
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
        f"*系統同時測量「傳統現況」、「字面數學重力」與「AI 脈絡靈魂」，以揭示理論的真實維度破缺。*\n"
        f"* 🗺️ **學術地圖現況 (Baseline)**：`[{data['baseline_hex']}]` (古典守成)\n"
        f"* 🧮 **數學尤拉指紋 (Math Gravity)**：`[{data['math_hex']}]`\n"
        f"* 🧠 **AI 靈魂指紋 (Neural Intent)**：`[{data['ai_hex']}]` - **{data['state_name']}**\n"
        f"* 🌌 **干涉解讀**：\n"
        f"  * 數學與 AI 意圖對比：**{math_ai_diff}**\n"
        f"  * 成功突破之傳統維度：**【{data['breakthrough_str']}】**\n\n"
        f"### 2. 🧮 尤拉相位觀測儀表板 (Euler Dashboard)\n"
        f"* **邏輯節點數**：`{data['vec_stats']['node_count']}` | **模長 (L2 Norm)**：`{data['vec_stats']['norm']:.8f}`\n"
        f"* **各維度字面引力對決**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"### 3. 🎯 狀態自鎖顯化摘要 (Auto-Locked Synthesis)\n"
        f"*(系統根據上述【三態共振】得出的突破維度與天命評語：「{data['state_desc']}」，強制引導 AI 寫出以下精準接地之論述：)*\n\n"
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
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 三態共振觀測</strong></p>\n"
        f"        <p>突破維度：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終演化狀態：[ {data['ai_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "        <p><em>V18.0 干涉儀協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open("WP_Ready_" + basename + ".html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    tex_content = data['full_text'].replace("#", "\\section")
    # 保護機制：若摘要極短，確保不會破壞 LaTeX 結構
    summary_text = data.get('summary', '摘要生成中...')
    if len(summary_text) > 200:
        summary_text = summary_text[:200] + "..."
        
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
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V18.0 三態共振防爆版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：三態干涉觀測軌跡\n")
        log_file.write("*本文件展示了最高維度的干涉儀邏輯：系統同時測量古典基準(Baseline)、字面數學引力(Math)、與高維上下文意圖(AI)。透過三者的差異，精準定位理論的突破點，並引導大腦產出不崩潰的完美顯化論述。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['ai_hex']
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
                
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data) # 恢復 LaTeX 匯出，修復 GitHub Action 錯誤！

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
