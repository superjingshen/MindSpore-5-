# Intelligent Triage System (MindSpore)

基于 Qwen2.5-7B 模型与 MindSpore 框架实现的智能医疗分诊系统。

**中文简体**

---

本项目是一个基于 **Qwen2.5-7B** 大语言模型微调的智能医疗分诊系统。利用 **MindSpore** 框架和 **MindNLP** 库，在 **Ascend NPU** 硬件上实现了高效的训练和推理。

## Features

1.  **国产算力适配:** 完全基于华为 Ascend NPU (910B) 和 MindSpore 框架开发。
2.  **高效微调:** 使用 PEFT/LoRA 技术进行参数高效微调 (Rank=8, Alpha=32)，大幅降低显存占用。
3.  **思维链推理 (CoT):** 模型输出完整的“诊疗推理”过程（症状理解 -> 患者画像 -> 系统定位 -> 科室匹配），覆盖 40 个标准门诊科室。
4.  **高性能推理:** 支持 LoRA 权重合并 (Merge) 与图编译预热，支持流式输出。
5.  **全流程覆盖:** 提供数据生成、模型训练、自动评估及 Web 演示的全套解决方案。

## Installation

### Tested Environments

*   Python 3.9
*   MindSpore 2.7.0 (Ascend)
*   MindNLP 0.5.1
*   Ascend 910B NPU

### Install Manually

#### Install Dependencies

```bash
pip install -r requirements.txt
```

## Data & Training

### Data Generation (Optional)

利用 DeepSeek API 自动构建高质量指令微调数据集。

```bash
python Data_Gen.py
```

### Training

使用 LoRA 技术在 NPU 上进行微调。

```bash
nohup python train_qwen_triage.py > train.log 2>&1 &
```

*   **Monitor:** `tail -f train.log`
*   **Output:** Loss 从 2.03 降至 1.40 左右。

## Start Inference

### Launch WebUI

启动基于 Gradio 的交互式 Web 界面。

```bash
python gradio_triage_app.py
```

访问地址: `http://0.0.0.0:7860`

### Evaluation

运行自动化测试脚本评估模型准确率。

```bash
python test_acc.py
```

## Performance

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | 95.00% | 测试集 100 题 (40 基础 + 60 拓展) |
| **Success Rate** | 100% | 无推理报错 |
| **Latency** | ~21s/题 | 流式输出，首字响应快 |
| **Coverage** | 100% | 覆盖 40 个标准门诊科室 |

## Directory Structure

| Name | Description |
| :--- | :--- |
| `Data_Gen.py` | 数据生成脚本 (DeepSeek API) |
| `train_qwen_triage.py` | 模型训练脚本 (MindNLP + LoRA) |
| `reasoning.py` | 推理核心类 (加载模型、推理逻辑) |
| `test_acc.py` | 自动化测试与评估脚本 |
| `gradio_triage_app.py` | Web 演示应用 |
| `requirements.txt` | 项目依赖 |
