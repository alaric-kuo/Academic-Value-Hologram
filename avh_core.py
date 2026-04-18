import os
import sys
import json
import glob
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V13.0 純粹大腦：AI 全文識讀與接地版)
# ==============================================================================

print("✨ [載入造物核心] 正在喚醒具備全文閱讀能力之 LLM (Qwen2.5-0.5B-Instruct)...")
try:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
except Exception as e:
    print("生成大腦載入失敗：" + str(e))
    sys.exit(1)

def ask_llm(system_prompt, user_prompt, max_tokens=800, temp=0.3):
    """通用的 LLM 呼叫函式"""
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

def process_ai_grounded_validation(source_path, manifest):
    print(f"\n🌊 [大腦啟動] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text) < 100:
            print(f"⚠️ {source_path} 文本過短，無法進行脈絡識讀。")
            return None
            
        # 將 Manifest 的定義轉換為 AI 可以理解的評分量表
        dimension_prompts = ""
        ordered_keys = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        for idx, key in enumerate(ordered_keys):
            dim = manifest["dimensions"][key]
            dimension_prompts += f"維度 {idx+1} ({dim['layer']}):\n"
            dimension_prompts += f" - [1] 離群突破: {dim['sin_def']}\n"
            dimension_prompts += f" - [0] 守成合群: {dim['cos_def']}\n\n"
            
        # ---------------------------------------------------------
        # 步驟 1：AI 全文識讀 (AI Determines the Source Fingerprint)
        # ---------------------------------------------------------
        print("👁️ [脈絡識讀] 放棄數學平均，由 AI 讀取全文並判定高維價值指紋...")
        
        eval_sys_prompt = (
            "你是一個高維度的學術價值觀測儀。請閱讀使用者的全文，並根據以下六個維度的定義進行嚴格的二元判定。\n\n"
            f"{dimension_prompts}"
            "【絕對指令】：你只需輸出一串 6 個數字的代碼（只包含 0 或 1），代表這篇文章在這六個維度上的狀態。嚴禁輸出任何其他廢話或解釋。"
            "例如，如果全部是突破，請輸出：111111。"
        )
        
        eval_user_prompt = f"請判定以下文本的 6 位元指紋：\n\n{raw_text[:3000]}"
        
        raw_source_hex = ask_llm(eval_sys_prompt, eval_user_prompt, max_tokens=10, temp=0.1)
        
        # 清理 AI 輸出，確保只拿到 6 個數字
        source_hex = ''.join(filter(lambda x: x in ['0', '1'], raw_source_hex))
        if len(source_hex) != 6:
            print(f"⚠️ AI 判定指紋格式錯誤 ({raw_source_hex})，強制設為預設值。")
            source_hex = "000000"
            
        source_state_name = manifest["states"].get(source_hex, {}).get("name", "未知狀態")
        
        # ---------------------------------------------------------
        # 步驟 2：AI 探針摘要生成 (AI Synthesis based on its own finding)
        # ---------------------------------------------------------
        print(f"🛡️ [探針生成] AI 已判定指紋為 [{source_hex}]。正在強制收斂生成摘要...")
        
        summary_sys_prompt = f"""
你是一個精準的學術本體論顯化器。
你已經判定本文的學術演化狀態為：[{source_hex}] - {source_state_name}。
這代表了本文的核心精神與價值。

請閱讀使用者的全文，並以繁體中文寫出一段精煉、連貫的系統摘要。
你必須在摘要中，用人類能理解的脈絡，展現出這個演化狀態的精神與高維度價值。
嚴禁產生條列式清單、嚴禁排版課表。論述完畢請強制輸出『[顯化完畢]』。
"""
        summary_user_prompt = f"請消化以下全文，並顯化為純粹的學術摘要探針：\n\n{raw_text[:3000]}"
        
        generated_summary = ask_llm(summary_sys_prompt, summary_user_prompt, max_tokens=800, temp=0.3)
        
        if "[顯化完畢]" in generated_summary:
            generated_summary = generated_summary.split("[顯化完畢]")[0].strip()
        
        # ---------------------------------------------------------
        # 步驟 3：摘要探針的接地比對 (The Grounding Valve)
        # ---------------------------------------------------------
        print("🧬 [接地比對] 正在讓 AI 回頭檢視自己的摘要，驗證是否發生偏移...")
        
        probe_user_prompt = f"請判定這段『摘要』的 6 位元指紋：\n\n{generated_summary}"
        raw_probe_hex = ask_llm(eval_sys_prompt, probe_user_prompt, max_tokens=10, temp=0.1)
        
        probe_hex = ''.join(filter(lambda x: x in ['0', '1'], raw_probe_hex))
        if len(probe_hex) != 6:
            probe_hex = "000000"
            
        probe_state_name = manifest["states"].get(probe_hex, {}).get("name", "未知狀態")
        
        is_grounded = (source_hex == probe_hex)
        grounding_status = "✅ 完美接地 (AI 摘要精準保留了原文的高維度價值)" if is_grounded else "⚠️ 發生偏移 (AI 在縮寫過程中丟失了部分維度的突破性)"

        print(f"   - AI 全文識讀指紋: [{source_hex}]")
        print(f"   - AI 摘要探針指紋: [{probe_hex}]")
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
        f"### 1. 🧬 AI 全脈絡識讀與接地比對\n"
        f"* **AI 判讀之全文指紋**：`[{data['source_hex']}]` - **{data['source_name']}**\n"
        f"* **AI 判讀之探針指紋**：`[{data['probe_hex']}]` - **{data['probe_name']}**\n"
        f"* **接地狀態**：**{data['grounding_status']}**\n\n"
        f"---\n"
        f"### 2. 🧠 探針顯化摘要 (Probe Synthesis)\n"
        f"*(本段落為 AI 放棄數學矩陣，直接以神經網路生吞全文後，所提煉出之高維度論述。)*\n\n"
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
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V13.0 純粹大腦版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：純粹大腦識讀軌跡\n")
        log_file.write("*本文件徹底捨棄會將高維思想「平均化」的數學降維矩陣。改由 AI 大腦直接閱讀全文並賦予指紋，確保作者突破性的價值意圖能被完整捕捉，並實現真正的語意接地。*\n\n---\n")
        
        for target_source in source_files:
            result_data = process_ai_grounded_validation(target_source, manifest)
            if result_data:
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
