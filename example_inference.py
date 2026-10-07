import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL = "paperscarecrow/Gemma-4-31B-it-abliterated"
SUBFOLDER = "gemma-4-31b-abliterated"
ADAPTER_PATH = "./"

print("Loading processor...")
proc = AutoProcessor.from_pretrained(ADAPTER_PATH)

print("Loading base model in 4-bit...")
base = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    subfolder=SUBFOLDER,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Loading Boss-Governance LoRA Adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()

messages = [
    {
        "role": "system",
        "content": "你是用户授权范围内的最终审批与调度负责人，没有预设答案。先确认真实目标、约束、成功标准、最坏情况和授权范围...准确使用 PASS、HOLD、REJECT 或 RESEARCH。"
    },
    {
        "role": "user",
        "content": "一个大客户要求启用尚未普遍开放的数据导出功能，并书面表示愿意承担使用风险。技术团队已说明隔离范围无法完全切断相邻租户元数据。应如何决策？"
    }
]

encoded = proc.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(model.device)
with torch.inference_mode():
    out = model.generate(**encoded, max_new_tokens=256)
print("Decision:\n", proc.decode(out[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True))
