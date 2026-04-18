import os
import sys
import json
import glob
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V12.0 純粹接地：摘要探針與原波包印證版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路 (paraphrase-multilingual-MiniLM)...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print("模型載入失敗：" + str(e))
    sys.exit(1)

print("✨ [載入造物核心] 正在喚醒具備收斂意志之 LLM (Qwen2.5-0.5B-Instruct)...")
try:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
except Exception as e:
    print("生成大腦載入失敗：" + str(e))
    sys.exit(1)

def measure_avh_hex(vector, manifest):
    """計算給定向量在 AVH 矩陣中的六十四卦指紋"""
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    hex_code = ""
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        sim_sin = np.dot(vector, v_sin) / (np.linalg.norm(vector) * np.linalg.norm(v_sin))
        sim_cos = np.dot(vector, v_cos) / (np.linalg.norm(vector) * np.linalg.norm(v_cos))
        hex_code += "1" if sim_sin > sim_cos else "0"
    return hex_code

def process_grounded_validation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            print(f"⚠️ {source_path} 文本結構過於單一，資訊熵不足。")
            return None
            
        # ---------------------------------------------------------
        # 步驟 1：測量原文直接指紋 (Source Fingerprint)
        # ---------------------------------------------------------
        print("🕸️ [原文測量] 計算全域語意質心與直接指紋...")
        embeddings = embedding_model.encode(paragraphs)
        source_psi = np.mean(embeddings, axis=0)
        source_hex = measure_avh_hex(source_psi, manifest)
        source_state_name = manifest["states"].get(source_hex, {}).get("name", "未知狀態")
        
        # ---------------------------------------------------------
        # 步驟 2：AI 探針摘要生成 (Probe Generation)
        # ---------------------------------------------------------
        print(f"🛡️ [探針生成] 指導 AI 根據直接指紋 [{source_hex}] 進行摘要收斂...")
        
        system_prompt = f"""
你是一個精準的學術本體論顯化器。
經過系統測量，本文的「學術演化狀態」為：[{source_hex}] - {source_state_name}。
這代表了本文的核心精神與突破方向。

請閱讀使用者的全文，並以繁體中文寫出一段精煉、連貫的系統摘要。
你必須在摘要中，精準地捕捉並展現上述演化狀態的精神。
嚴禁產生條列式清單、嚴禁排版課表。論述完畢請強制輸出『[顯化完畢]』。
"""
        user_prompt = f"請消化以下全文，並顯化為純粹的學術摘要探針：\n\n{raw_text[:2500]}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
        
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=800,
            temperature=0.3, 
            repetition_penalty=1.15
        )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        raw_generated_summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if "[顯化完畢]" in raw_generated_summary:
            raw_generated_summary = raw_generated_summary.split("[顯化完畢]")[0].strip()
        
        generated_summary = zhconv.convert(raw_generated_summary, 'zh-tw')
        
        # ---------------------------------------------------------
        # 步驟 3：摘要探針的接地比對 (The Grounding Valve)
        # ---------------------------------------------------------
        print("🧬 [接地比對] 正在測量 AI 摘要探針的指紋，驗證是否完美接地...")
        summary_vec = embedding_model.encode([generated_summary])[0]
        probe_hex = measure_avh_hex(summary_vec, manifest)
        probe_state_name = manifest["states"].get(probe_hex, {}).get("name", "未知狀態")
        
        # 判斷是否接地 (指紋是否一致)
        is_grounded = (source_hex == probe_hex)
        grounding_status = "✅ 完美接地 (探針指紋與原文絕對吻合)" if is_grounded else "⚠️ 發生偏移 (AI 摘要未能精準鎖定原文維度)"

        print(f"   - 原文指紋: [{source_hex}]")
        print(f"   - 探針指紋: [{probe_hex}]")
        print(f"   - 狀態: {grounding_status}")

        return {
            "source_hex": source_hex,
            "source_name": source_state_name,
            "probe_hex": probe_hex,
            "probe_name": probe_state_name,
            "is_grounded": is_grounded,
            "grounding_status": grounding_status,
            "summary": generated_summary,
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 邏輯顯化過程錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    
    log_output = (
        f"## 📡 演化顯化軌跡：`{target_file}`\n"
        f"* **物理時間戳**：`{timestamp}`\n\n"
        f"### 1. 🧬 邏輯接地比對 (Self-Consistency Validation)\n"
        f"* **原文直接指紋**：`[{data['source_hex']}]` - **{data['source_name']}**\n"
        f"* **AI 探針指紋**：`[{data['probe_hex']}]` - **{data['probe_name']}**\n"
        f"* **接地狀態**：**{data['grounding_status']}**\n\n"
        f"---\n"
        f"### 2. 🧠 探針顯化摘要 (Probe Synthesis)\n"
        f"*(本段落為 AI 消化全文後生成的摘要。透過反向計算此摘要的指紋，系統得以驗證 AI 是否產生語意幻覺。)*\n\n"
        f"> **{data['summary']}**\n\n"
        f"---\n"
    )
    return log_output

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
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V12.0 純粹接地版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：邏輯接地驗證軌跡\n")
        log_file.write("*本文件紀錄了最純粹的「一石二鳥」邏輯：系統先算出原文的直接指紋，再讓 AI 寫出摘要。最後，系統計算 AI 摘要的指紋進行比對。唯有兩者完全吻合，才代表 AI 真正讀懂了文本，實現了完美的邏輯接地。*\n\n---\n")
        
        for target_source in source_files:
            result_data = process_grounded_validation(target_source, manifest)
            if result_data:
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
