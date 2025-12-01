import os
import json
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import ops
import gc 

# 1. 极限显存压榨：允许最大用到 31GB (32GB卡保留1GB给系统足够了)
os.environ['MS_ALLOC_CONF'] = 'enable_vmm:True,vmm_align_size:2MB,max_device_memory:31GB'

from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.peft import LoraConfig, TaskType, get_peft_model

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            o = json.loads(s)
            yield o.get("instruction", ""), o.get("input", ""), o.get("output", "")

def build_processor(tokenizer, max_length):
    def proc(instruction, _input, output):
        msg = [{"role": "user", "content": (instruction + "\n" + _input).strip()}]
        prompt_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        response_text = f"{output}{tokenizer.eos_token}"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids
        
        # 仅截断超长数据
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        
        return (
            np.array(input_ids, dtype=np.int32),
            np.array(attention_mask, dtype=np.int32),
            np.array(labels, dtype=np.int32),
        )
    return proc

def make_dataset(tokenizer, path, max_length, batch_size, num_workers):
    proc = build_processor(tokenizer, max_length)
    def generator():
        for instruction, _input, output in load_jsonl(path):
            yield proc(instruction, _input, output)
    dataset = ds.GeneratorDataset(generator, column_names=["input_ids", "attention_mask", "labels"], num_parallel_workers=num_workers, shuffle=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="/home/ma-user/work/dataset/Triage_Data.jsonl")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--output_dir", default="/home/ma-user/work/output/qwen_triage")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    # 策略：保持1024，靠异常捕获机制来跳过个别OOM的数据
    # ap.add_argument("--max_length", type=int, default=1024) 
    ap.add_argument("--max_length", type=int, default=900) 
    ap.add_argument("--dtype", default="fp16") 
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cache_dir", default="/home/ma-user/work/model")
    args = ap.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    
    # NPU可用性检查
    try:
        _ = ms.Tensor([1.0], dtype=ms.float32).to('npu:0')
    except:
        pass # 忽略这里的报错，后续会自动处理

    dtype = ms.bfloat16 if args.dtype == "bf16" else ms.float16
    os.makedirs(args.cache_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, ms_dtype=dtype, cache_dir=args.cache_dir)
    dataset = make_dataset(tokenizer, args.dataset_path, args.max_length, args.batch_size, args.num_workers)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, ms_dtype=dtype, cache_dir=args.cache_dir, use_cache=False)
    print("正在将模型移动到 NPU...")
    model = model.to('npu:0')
    model.gradient_checkpointing_enable()
    
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()], inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model = get_peft_model(model, cfg)
    model.set_train()
    
    trainable_params = list(model.trainable_params())
    lr_tensor = ms.Tensor(args.learning_rate, dtype=ms.float32)

    # 训练步函数
    def train_step(input_ids, attention_mask, labels):
        def forward_fn():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return output.loss
        
        grad_fn = ms.value_and_grad(forward_fn, None, tuple(trainable_params))
        loss, grads = grad_fn()
        
        for param, grad in zip(trainable_params, grads):
            if grad is not None:
                p_data = param.data if hasattr(param, 'data') else param
                g_cast = grad.astype(p_data.dtype)
                lr_cast = lr_tensor.astype(p_data.dtype)
                ops.assign_sub(p_data, ops.mul(g_cast, lr_cast))
        return loss
    
    out_dir = os.path.join(args.output_dir, args.dtype) 
    os.makedirs(out_dir, exist_ok=True)
    
    step = 0
    print(f"[INFO] 开始训练 Loop (已开启OOM熔断保护)...")
    for epoch in range(args.epochs):
        for batch in dataset.create_tuple_iterator(output_numpy=True):
            # 将 numpy 数据转为 Tensor 并移动到 NPU
            input_ids = ms.Tensor(batch[0], dtype=ms.int32).to('npu:0')
            attention_mask = ms.Tensor(batch[1], dtype=ms.int32).to('npu:0')
            labels = ms.Tensor(batch[2], dtype=ms.int32).to('npu:0')
            
            # ===============================================
            # OOM 捕获与跳过
            # ===============================================
            try:
                loss = train_step(input_ids, attention_mask, labels)
                step += 1
                print(f"[训练] Epoch {epoch+1}, Step {step}, Loss: {loss.asnumpy():.4f}")
                
                if args.save_steps and step % args.save_steps == 0:
                    model.save_pretrained(os.path.join(out_dir, f"checkpoint-{step}", "adapter_model"), safe_serialization=True)
            
            except RuntimeError as e:
                err_str = str(e)
                # 捕获常见的显存相关错误
                if 'Alloc failed' in err_str or 'Mapped too much memory' in err_str or 'Out of memory' in err_str:
                    # 打印醒目的警告
                    print(f"\n{'='*50}")
                    print(f"[WARNING] ⚠️ 触发显存保护机制 (Step {step+1})")
                    print(f"[INFO] 跳过当前数据: 长度 {input_ids.shape[1]}")
                    print(f"{'='*50}\n")
                    
                    # 只有在这里，我们才执行 Python 垃圾回收
                    del input_ids, attention_mask, labels
                    gc.collect()
                    continue # 跳过本次循环，不退出程序！
                else:
                    # 如果是其他代码错误，则正常报错
                    raise e
            
            if args.max_steps and step >= args.max_steps: break
        if args.max_steps and step >= args.max_steps: break
    
    print(f"[INFO] 训练完成！")
    model.save_pretrained(os.path.join(out_dir, "final", "adapter_model"), safe_serialization=True)
    print(f"[INFO] 模型保存完成")

if __name__ == "__main__":
    main()