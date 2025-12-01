import argparse
import json
import os
import random
import re
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Set

# 第三方库检查
try:
    import openai
    from tqdm import tqdm
except ImportError:
    print("错误: 缺少必要库。请运行: pip install openai tqdm")
    sys.exit(1)

# --- 全局正则 ---
RE_SPLIT_DEPT = re.compile(r"\n(?=[\u4e00-\u9fff]+(?:科门诊|门诊|科|中心))")
RE_SYMPTOM = re.compile(r".*?症状[：:]")
RE_DISEASE = re.compile(r".*?疾病[：:]")
RE_KEYWORD = re.compile(r".*?关键词[：:]")
RE_EMERGENCY = re.compile(r".*?急诊[：:]")
RE_SPLIT_ITEMS = re.compile(r"[、，,；;]")

def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

def parse_departments(content: str) -> Dict[str, Dict[str, List[str]]]:
    titles = {"内科", "外科", "妇产科", "儿科", "专科门诊", "感染性疾病科", "骨外科"}
    sections = RE_SPLIT_DEPT.split(content)
    result: Dict[str, Dict[str, List[str]]] = {}
    
    for section in sections:
        s = section.strip()
        if not s: continue
        lines = s.split("\n")
        name = lines[0].strip()
        if not name or name in titles: continue
        
        data = {
            "symptoms": [], "diseases": [], "keywords": [], 
            "emergency": [], "full_text": ""
        }
        data["full_text"] = s.split("\n", 1)[1] if "\n" in s else s
        
        for line in lines[1:]:
            line = line.strip()
            target = None
            text = ""
            if "症状" in line:
                target = data["symptoms"]
                text = RE_SYMPTOM.sub("", line)
            elif "疾病" in line:
                target = data["diseases"]
                text = RE_DISEASE.sub("", line)
            elif "关键词" in line:
                target = data["keywords"]
                text = RE_KEYWORD.sub("", line)
            elif "急诊" in line:
                target = data["emergency"]
                text = RE_EMERGENCY.sub("", line)
            
            if target is not None:
                items = [x.strip() for x in RE_SPLIT_ITEMS.split(text) if x.strip()]
                target.extend(items)
        result[name] = data
    return result

def build_question_prompt(name: str, data: Dict[str, List[str]]) -> List[Dict[str, str]]:
    sx = "、".join(data.get("symptoms", [])[:10]) or "相关症状"
    dx = "、".join(data.get("diseases", [])[:5]) or "相关疾病"
    types = ["症状描述型", "疾病咨询型", "复诊咨询型", "预防咨询型", "特殊人群型", "科室选择型", "复合症状型"]
    qtype = random.choice(types)
    specifics = {
        "症状描述型": f"围绕{sx}中的1-2个症状，描述具体时间线与影响",
        "疾病咨询型": f"结合{dx}中的某疾病，询问当前处理与就诊建议",
        "复诊咨询型": "描述既往就诊与当前复发或随访问题",
        "预防咨询型": "询问预防与日常管理建议，场景真实",
        "特殊人群型": "儿童/孕妇/老人等人群的差异化咨询",
        "科室选择型": "对比两个可能科室，询问该科是否更合适",
        "复合症状型": "包含两类以上症状的关联与就医疑惑",
    }
    system = (
        f"根据以下信息生成一个患者咨询问题。\n科室：{name}\n常见症状：{sx}\n常见疾病：{dx}\n"
        f"类型：{qtype}\n要求：{specifics.get(qtype, '口语化自然，仅返回患者问题')}。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "生成一个问题"}]

def build_answer_prompt(name: str, data: Dict[str, List[str]], question: str, canonical: str, variants: List[str], max_chars: int) -> List[Dict[str, str]]:
    base = data.get('full_text', '')
    snippet = ""
    if base:
        start = 0
        if len(base) > max_chars:
            start = random.randint(0, max(0, len(base) - max_chars))
        snippet = base[start:start+max_chars] + ('...' if start + max_chars < len(base) else '')
    
    ctx_str = f"科室：{name}\n原始科室段落（截断）：\n{snippet}\n" if snippet else f"科室：{name}\n（无详细资料）"
    
    group_note = ""
    if len(variants) > 1:
        others = [v.replace(canonical, '') for v in variants if v != canonical]
        others_clean = [o.strip("（） ") for o in others if o.strip("（） ")]
        if others_clean:
            group_note = f"（注：医院挂号系统包含{canonical}、{'、'.join(others_clean)}等分组，任选其一即可）"
            
    # --- 核心修改：引入 CoT 提示词 ---
    system = (
        "你是经验丰富的医疗分诊专家。请根据患者咨询和科室信息，给出包含详细推理过程的专业建议。\n"
        "请严格按照以下 Markdown 格式结构输出：\n\n"
        "### 诊疗推理\n"
        "1. **症状理解**：[简述患者核心症状及病程]\n"
        "2. **患者画像**：[提取年龄、性别、特殊人群特征等]\n"
        "3. **系统定位**：[分析症状涉及的生理系统]\n"
        "4. **科室匹配**：[结合科室诊疗范围进行匹配]\n"
        "5. **紧急程度评估**：[评估风险等级：低/中/高]\n\n"
        "### 推荐科室\n"
        f"建议挂**{canonical}**门诊号{group_note}。\n\n"
        "### 医生建议\n"
        "[针对患者问题的具体医疗建议，3-4条]\n\n"
        "### 安全提示\n"
        "以上信息仅供参考，不能替代专业医疗诊断。如出现急症请立即前往急诊。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": f"患者咨询：{question}\n科室信息：\n{ctx_str}"}]

def validate_answer(answer: str, target_canonical: str, all_names_set: Set[str], v2c: Dict[str, str], allowed_variants: List[str]) -> bool:
    if not answer: return False
    
    # 1. 检查是否包含核心结构 "诊疗推理"
    if "诊疗推理" not in answer or "推荐科室" not in answer:
        return False
        
    # 2. 检查是否推荐了正确的科室 (在整个回答中出现即可，CoT会让模型多次提及)
    # 严格模式：检查 "### 推荐科室" 后面的内容是否包含目标科室，防止推理中提到但结论没推荐
    try:
        recommend_section = answer.split("推荐科室")[1].split("医生建议")[0]
        hit = (target_canonical in recommend_section) or any(v in recommend_section for v in allowed_variants)
        return hit
    except IndexError:
        # 如果切分失败，退回到全文搜索
        hit = (target_canonical in answer) or any(v in answer for v in allowed_variants)
        return hit

def precompute_canonical_data(groups: Dict[str, List[str]], dep: Dict[str, Dict]) -> Dict[str, Dict]:
    cache = {}
    for canonical, variants in groups.items():
        agg = {"symptoms": [], "diseases": [], "keywords": [], "emergency": [], "full_text": ""}
        seen = {k: set() for k in agg}
        if dep.get(canonical, {}).get("full_text"):
            agg["full_text"] = dep[canonical]["full_text"]
        for v in (variants or [canonical]):
            d = dep.get(v, dep.get(canonical, {}))
            if not agg["full_text"] and d.get("full_text"):
                agg["full_text"] = d["full_text"]
            for k in ["symptoms", "diseases", "keywords", "emergency"]:
                for item in d.get(k, []):
                    if item and item not in seen[k]:
                        seen[k].add(item)
                        agg[k].append(item)
        cache[canonical] = agg
    return cache

def worker_thread(
    thread_id: int,
    client: openai.OpenAI,
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    model: str,
    temp: float,
    max_tokens: int,
    max_chars: int,
    all_names_set: Set[str],
    v2c: Dict[str, str],
    groups: Dict[str, List[str]],
    canon_cache: Dict[str, Dict],
    dep: Dict[str, Dict],
    max_retries: int
):
    time.sleep(random.uniform(0, 2.0))
    
    while True:
        try:
            item = task_queue.get(timeout=2)
            if item is None: break
            dept_name, retry_count = item
        except queue.Empty:
            break 

        if retry_count > max_retries:
            print(f"!! [放弃] {dept_name}: 失败次数过多")
            result_queue.put({"status": "failed", "dept": dept_name})
            task_queue.task_done()
            continue

        try:
            canon_data = canon_cache.get(dept_name)
            vars_list = groups.get(dept_name, [dept_name])
            ask_name = random.choice(vars_list) if vars_list else dept_name
            
            # 生成问题
            q_msgs = build_question_prompt(ask_name, dep.get(ask_name, canon_data))
            resp_q = client.chat.completions.create(
                model=model, messages=q_msgs, temperature=temp, max_tokens=max_tokens
            )
            question = resp_q.choices[0].message.content

            # 生成 CoT 回答 (注意 max_tokens 可能需要调大，建议外部传入 1024)
            a_msgs = build_answer_prompt(ask_name, canon_data, question, dept_name, vars_list, max_chars)
            resp_a = client.chat.completions.create(
                model=model, messages=a_msgs, temperature=temp, max_tokens=max_tokens * 2 # 回答变长了，临时给更多token
            )
            answer = resp_a.choices[0].message.content
            
            # 校验
            if validate_answer(answer, dept_name, all_names_set, v2c, vars_list):
                result = {
                    "status": "success",
                    "data": {
                        "instruction": "请根据患者症状推荐合适的科室，并给出详细的诊疗思路。", # 指令更新
                        "input": question,
                        "output": answer,
                        "metadata": {"department": dept_name}
                    }
                }
                result_queue.put(result)
            else:
                # print(f"-> [重试] {dept_name}: 格式或科室校验失败")
                task_queue.put((dept_name, retry_count + 1))
        
        except Exception as e:
            # print(f"[T{thread_id}] 异常: {str(e)[:50]}")
            time.sleep(1)
            task_queue.put((dept_name, retry_count + 1))
        finally:
            task_queue.task_done()

def run(args: argparse.Namespace) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or args.api_key
    base_url = os.getenv("DEEPSEEK_BASE_URL") or args.base_url
    
    try:
        from model_key import API_KEY as FILE_API_KEY, BASE_URL as FILE_BASE_URL
        api_key = api_key or FILE_API_KEY
        base_url = base_url or FILE_BASE_URL
    except ImportError:
        pass

    if not api_key:
        print("错误: 未找到 API Key")
        return 2

    client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=60.0)

    src = Path(args.departments_file)
    if not src.exists():
        print(f"文件不存在: {src}")
        return 2
    
    print(f"正在解析: {src.name} ...")
    dep = parse_departments(load_text(src))
    names = list(dep.keys())
    
    groups = {}
    v2c = {}
    for n in names:
        c = n
        m1 = re.match(r"^(.*?门诊)\s*\d+\s*组$", n)
        m2 = re.match(r"^(.*?门诊)（.+?）$", n)
        m3 = re.match(r"^(骨外科)\s*\d+\s*组$", n)
        if m1: c = m1.group(1)
        elif m2: c = m2.group(1)
        elif m3: c = m3.group(1)
        v2c[n] = c
        groups.setdefault(c, []).append(n)
        
    canonical_names = list(groups.keys())
    print("预计算科室上下文...")
    canon_cache = precompute_canonical_data(groups, dep)
    all_names_set = set(names)

    target_total = args.samples
    task_queue = queue.Queue()
    
    if args.distribution == "balanced":
        q, r = divmod(target_total, len(canonical_names))
        base_tasks = canonical_names * q + random.sample(canonical_names, r)
        random.shuffle(base_tasks)
        for t in base_tasks: task_queue.put((t, 0))
    else:
        for _ in range(target_total):
            task_queue.put((random.choice(canonical_names), 0))

    concurrency = min(args.concurrency, target_total, 50)
    print(f"任务总数: {target_total} | 并发: {concurrency}")

    result_queue = queue.Queue()
    threads = []
    for i in range(concurrency):
        t = threading.Thread(
            target=worker_thread,
            args=(
                i, client, task_queue, result_queue, 
                args.model, args.temperature, args.max_tokens, 
                args.max_context_chars, all_names_set, v2c, groups, 
                canon_cache, dep, 5
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_questions = set()
    processed_count = 0
    
    pbar = tqdm(total=target_total, desc="生成进度", unit="条")
    
    with out_path.open("w", encoding="utf-8") as f:
        while processed_count < target_total:
            try:
                res = result_queue.get(timeout=1)
                
                if res["status"] == "failed":
                    pbar.write(f"跳过: {res['dept']}")
                    processed_count += 1
                    pbar.update(1)
                    continue

                item = res["data"]
                q_text = item["input"].strip()
                
                if q_text in seen_questions:
                    task_queue.put((item["metadata"]["department"], 0))
                    continue
                
                seen_questions.add(q_text)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                processed_count += 1
                pbar.update(1)
                
            except queue.Empty:
                if task_queue.empty() and result_queue.empty() and processed_count < target_total:
                    time.sleep(1)
                    continue
            except KeyboardInterrupt:
                break

    pbar.close()
    print(f"\n完成！数据已保存: {out_path}")
    return 0

def main():
    parser = argparse.ArgumentParser()
    default_dept = Path("dataset/整理后的科室信息.txt")
    parser.add_argument("--departments-file", default=str(default_dept))
    # --- 修改默认输出文件名 ---
    parser.add_argument("--output", default="/home/ma-user/work/dataset/Triage_Data2.jsonl")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    # --- 增加 Token 限制以适应长回答 ---
    parser.add_argument("--max-tokens", type=int, default=800) 
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--concurrency", type=int, default=50) 
    parser.add_argument("--distribution", default="balanced", choices=["balanced", "random"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)
    sys.exit(run(args))

if __name__ == "__main__":
    main()