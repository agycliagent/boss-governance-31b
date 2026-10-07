# 🏛️ Boss-Governance-31B (老板治理与调度中枢大模型)

[![Model Size](https://img.shields.io/badge/Model_Size-31B_Dense-blue.svg)](https://huggingface.co/paperscarecrow/Gemma-4-31B-it-abliterated)
[![Base Model](https://img.shields.io/badge/Base_Model-Gemma--4--31B--it--abliterated-orange.svg)](https://huggingface.co/paperscarecrow/Gemma-4-31B-it-abliterated)
[![Technique](https://img.shields.io/badge/Technique-4--bit_NF4_QLoRA-purple.svg)](https://github.com/agycliagent/boss-governance-31b)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**Boss-Governance-31B** 是一个专门针对**复杂系统治理、任务拆解与终极审批调度（Management & Dispatch Boss）**而深度调优的 31B 稠密参数大模型。

不同于传统被动的聊天应答或工具执行模型，本模型作为**全局决策主脑**，在专家推理、团队汇报与外部输入面前承担独立把关、权责划分与调度审批职责。

---

## 🚀 核心基因与去审查特征 (Base Model & De-censorship)

- **底模基座**：`paperscarecrow/Gemma-4-31B-it-abliterated` (Commit `2491ee58472b82a2372d288f982943b9d5e6a82c`)。
- **彻底去敏与零机械拒答 (Abliterated & Uncensored)**：
  - 传统 Google 官方原生模型存在过度严苛的安全护栏，经常在面对正常的系统运维、权限重构、故障排查或应急调度等敏感指令时产生“机械拒答（Refusal）”。
  - 本模型采用社区深度脱敏与去审查底模（Abliterated Base），从底层移除了机械安全锁。
  - **绝不莫名拒答、绝不因敏感词退缩**，专注于执行最高层级的架构治理与客观业务研判。

---

## 🎯 决策四态规范 (4-State Governance Paradigm)

在任何方案审批与权限调用场景中，模型严格落地“老板四态裁决规范”：

1. **`PASS`（放行审批）**：方案目标清晰、因果链完备、回滚与停止边界具备客观验证，全权放行并派发下一步；
2. **`HOLD`（暂缓挂起）**：发现逻辑缺失、跨模块接口未经论证或存在越权风险，要求提出方补全必要证据，坚决不盲目放行；
3. **`REJECT`（坚决驳回）**：触犯核心红线、越权或目标硬失败，立即驳回并给出明确违规事实；
4. **`RESEARCH`（专项调查）**：关键事实存在未知或数据冲突，指派最低充分资源进行定向事实深挖。

### 核心管理原则：
- **审批责任不可让渡**：专家负责专业推理，主脑负责独立审批；不因专家资历、多数意见或排行榜推荐动摇判断。
- **杜绝事后脑补**：关键逻辑不清由提出者解释，严禁模型事后自行替方案“打补丁”。

---

## 📊 训练规格与收敛指标 (Training & Benchmarks)

- **训练环境**：NVIDIA A100-SXM4-80GB (High-RAM 实例)
- **量化与算子**：BitsAndBytes 4-bit NF4 量化加载，bfloat16 计算精度，Paged AdamW 优化器
- **微调架构**：全线性层 QLoRA（q, k, v, o, gate, up, down 投影层），可训参数量 1.22 亿（0.3898%）
- **数据保真**：500 题黄金 SFT 治理全集，经由 SHA256 哈希硬锁（`e70ccbad7d9c...`）预检准入，确保零错喂、零污染
- **验证表现**：
  - 训练步数：66 Steps（1 个完整 Epoch）
  - 单步 Loss：从初始 **8.029** 平滑收敛至 **2.940**（平均训练 Loss **4.124**）
  - 验证集评估：104 题独立全量高难度验证集测得 **Eval Loss: 3.221**
  - 显存控制：峰值稳固压制在 **29.75 GB**，全程零 OOM

---

## 💻 快速上手 (Quickstart)

```python
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "paperscarecrow/Gemma-4-31B-it-abliterated"
SUBFOLDER = "gemma-4-31b-abliterated"
ADAPTER_PATH = "./boss_adapter" # 从 Release 解压

# 1. 4-bit 加载底模
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    subfolder=SUBFOLDER,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 2. 挂载 Boss-Governance LoRA
model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

# 3. 构造治理请求
messages = [
    {
        "role": "system",
        "content": "你是用户授权范围内的最终审批与调度负责人，没有预设答案。先确认真实目标、约束、成功标准、最坏情况和授权范围...准确使用 PASS、HOLD、REJECT 或 RESEARCH。"
    },
    {
        "role": "user",
        "content": "三位专家分别证明了子模块可行，但未定义跨模块数据传输的安全接口。应如何判定？"
    }
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))
```

---

## 📦 权重下载 (Downloads)

| 资产 | 规格 | 下载途径 |
| :--- | :--- | :--- |
| **完整 LoRA 权重包** (`31B_final_adapter_v2.6.zip`) | 445 MB (解压后 541 MB) | [GitHub Releases v1.0.0 页面下载](../../releases) 或 [高速直链镜像](https://desk.xinjiyuan.tech/media/31B_final_adapter_v2.6.zip) |
| **配置文件与词表** | `adapter_config.json`, `tokenizer.json` 等 | 直接存放在本代码仓库根目录 |

---

## ⚖️ License

本项目微调权重遵循 Apache 2.0 许可证开源，基模权重遵循 Gemma 社区许可及去审查衍生说明。
