import os

# [性能优化] 开启大页内存，减少 NPU 通信开销
os.environ["MS_ALLOC_CONF"] = "enable_vmm:True,vmm_align_size:2MB"

# [离线配置]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import time
from threading import Thread
import mindspore as ms
from typing import Optional, List, Dict
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from mindnlp.peft import PeftModel

class Reasoner:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, device: str = "Ascend", dtype: str = "fp16", cache_dir: Optional[str] = None, device_id: int = 0):
        # 1. 设置上下文
        try:
            # 这里的 mode=ms.PYNATIVE_MODE 对于 LLM 生成任务兼容性比较好
            # 如果想极致加速，可以尝试 GRAPH_MODE，但动态 Shape 容易报错
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device, device_id=device_id)
            print(f"[Device] MindSpore Context: {device}:{device_id}")
        except Exception as e:
            print(f"Context 设置失败: {e}")
        
        # 2. Ascend 910A/B  float16 
        ms_dtype = ms.float16 if dtype == "fp16" else ms.bfloat16
        
        kwargs = {
            "ms_dtype": ms_dtype,
            "local_files_only": True,
        }
        
        # 路径解析逻辑
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            
        def _resolve_local_model_path(model_path: str, cache_dir: Optional[str]) -> str:
            if cache_dir and isinstance(model_path, str) and "/" in model_path:
                org, repo = model_path.split("/", 1)
                base_dir = os.path.join(cache_dir, f"models--{org}--{repo}", "snapshots")
                if os.path.isdir(base_dir):
                    candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                    for cand in sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True):
                        if os.path.exists(os.path.join(cand, "config.json")):
                            return cand
            return model_path

        local_model_path = _resolve_local_model_path(model_path, cache_dir)
        print(f"正在加载基座模型: {local_model_path} (FP16)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, **kwargs)
        base = AutoModelForCausalLM.from_pretrained(local_model_path, **kwargs)
        
        # ==================== [核心加速点 1]：合并 LoRA 权重 ====================
        if adapter_path and os.path.isdir(adapter_path):
            print(f"正在加载 LoRA 并进行权重合并 (Merge)...")
            try:
                # 加载 Peft 模型
                peft_model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
                
                # 【关键步骤】将 LoRA 权重彻底合并进基座模型
                # 这样推理时就变成了一个普通的模型，不需要计算 LoRA 的旁路，速度提升显著
                self.model = peft_model.merge_and_unload()
                print("LoRA 权重已合并，推理速度将最大化。")
            except Exception as e:
                print(f"LoRA 合并失败 (可能版本不支持)，回退到挂载模式: {e}")
                try:
                    self.model = PeftModel.from_pretrained(base, adapter_path)
                except:
                    self.model = base
        else:
            self.model = base
        
        print(f"正在将模型移动到 NPU:{device_id}...")
        self.model = self.model.to(f'npu:{device_id}')
        self.model.set_train(False) # 确保是 eval 模式
        self.device_id = device_id
        
        # ==================== [加速点 2]：图编译预热 ====================
        # MindSpore 需要编译算子图，第一次运行非常慢。
        # 我们进行一次“热身”，把编译时间消耗在初始化阶段。
        print("正在预热编译 (Warmup)...")
        try:
            dummy_input = self.tokenizer("Warmup", return_tensors="ms")
            dummy_input = {k: v.to(f'npu:{device_id}') for k, v in dummy_input.items()}
            # 强制执行一次生成
            self.model.generate(**dummy_input, max_new_tokens=1, do_sample=False)
            print("预热完成。")
        except Exception as e:
            print(f"预热失败: {e}")

    def _build_messages(self, user_text: str):
        # ... (保持原本的 Prompt 逻辑) ...
        departments = [
            "中医科门诊", "产科门诊", "介入血管科门诊", "儿科门诊", "儿童保健门诊",
            "全科医学科门诊", "内分泌门诊", "创面外科门诊", "医学美容科门诊", "口腔科门诊",
            "呼吸与危重症医学科门诊", "咳喘专病门诊", "哮喘专病门诊", "妇科门诊", "康复科门诊",
            "心血管内科门诊", "感染性疾病科门诊", "慢阻肺专病门诊", "新生儿门诊", "普通外科门诊",
            "泌尿外科门诊", "消化内科门诊", "消化外科门诊", "疼痛科门诊", "皮肤科门诊",
            "眼科门诊", "神经内科门诊", "神经外科门诊", "精神心理科门诊", "老年病科门诊",
            "耳鼻咽喉头颈外科门诊", "肛肠科门诊", "肾脏内科门诊", "肿瘤科、甲乳外科门诊", "胸外科门诊",
            "营养科门诊", "血液透析中心", "针灸科", "骨外科", "麻醉科门诊"
        ]
        
        sys = f"""你是医院手机预约挂号系统的智能分诊助手。患者正在通过手机 APP 预约门诊号，需要你根据症状描述推荐合适的科室。

**场景说明**：
- 这是预约挂号场景，患者会提前几天或当天预约门诊，不是急诊（有急事患者会直接拨打 120）
- 本院共有 40 个可挂科室，你只能从以下科室中推荐：
  {', '.join(departments)}

**输出要求**：
请按照以下 Markdown 结构输出专业的分诊建议：

### 诊疗推理
1. **症状理解**：[简述患者核心症状、病程、影响]
2. **患者画像**：[年龄、性别、特殊人群（孕妇/儿童/老人）特征]
3. **系统定位**：[判断症状涉及的生理系统或器官]
4. **科室匹配**：[结合上述分析，说明为何推荐该科室]

### 推荐科室
建议挂**[科室名称]**

### 预约建议
- 建议预约时间：[如：近期/本周内/尽快等]
- 就诊准备：[需要携带的检查报告、注意事项等，2-3 条]

### 安全提示
以上建议仅供参考。若症状急剧加重、出现剧烈疼痛、大出血等危急情况，请立即拨打 120 或前往急诊。"""
        
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_text},
        ]

    def _parse_json(self, text: str):
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            s = text[i : j + 1]
            try:
                return json.loads(s)
            except Exception:
                pass
        return {"raw": text}

    def generate(self, messages: List[Dict], temperature: float = 0.0, max_new_tokens: int = 512):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="ms")
        inputs = {k: v.to(f'npu:{self.device_id}') for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # ==================== [加速点 3]：生成参数微调 ====================
        try:
            t0 = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,       # 必须开启 KV Cache
                do_sample=False,      # 关闭采样，NPU 上 Greedy Search 最快
                temperature=None,     # 显式置空
                top_p=None
            )
            t1 = time.time()
            gen_len = outputs[0].shape[0] - input_len
            speed = gen_len / (t1 - t0)
            print(f"推理速度: {speed:.2f} tokens/s (总耗时: {t1 - t0:.2f}s)")
        except Exception as e:
            print(f"生成报错: {e}")
            return f"生成出错: {e}"
            
        if outputs[0].shape[0] > input_len:
            generated_ids = outputs[0][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return text

    # triage 和 reason 方法
    def triage(self, user_text: str, max_new_tokens: int = 512):
        msgs = self._build_messages(user_text)
        return self.generate(msgs, max_new_tokens=max_new_tokens)

    def stream_triage(self, user_text: str, max_new_tokens: int = 512):
        msgs = self._build_messages(user_text)
        yield from self.stream_generate(msgs, max_new_tokens=max_new_tokens)

    def stream_generate(self, messages: List[Dict], max_new_tokens: int = 512):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="ms")
        inputs = {k: v.to(f'npu:{self.device_id}') for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True,
            do_sample=False,
            temperature=None,
            top_p=None
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def reason(self, user_text: str, max_new_tokens: int = 512):
        msgs = self._build_messages(user_text)
        text = self.generate(msgs, max_new_tokens=max_new_tokens)
        return self._parse_json(text)