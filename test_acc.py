import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ================================================================
# 环境准备（MindSpore 2.7.0 + MindNLP 0.5.1 + Ascend 910B4）
# ================================================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MS_ENABLE_GE", "1")

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from reasoning import Reasoner  # noqa: E402


def load_test_data(jsonl_path: str) -> List[Dict]:
    """加载测试数据集（JSONL格式）"""
    test_cases = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))
    return test_cases


def load_test_data_from_txt(txt_path: str) -> List[Dict]:
    """从test.txt加载测试数据（纯文本格式）"""
    import re
    test_cases = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析格式：序号. 患者："问题内容"——答案：科室名称
    pattern = r'\d+\.\s*患者[：:][""](.+?)[""]——答案[：:](.+?)(?=\n|$)'
    matches = re.findall(pattern, content)
    
    for idx, (question, department) in enumerate(matches, start=1):
        test_cases.append({
            'input': question.strip(),
            'output': '',  # test.txt没有完整的期望输出
            'metadata': {
                'department': department.strip()
            }
        })
    
    return test_cases


def extract_department(text: str) -> str:
    """从输出文本中提取科室名称"""
    # 方法1: 查找 **科室名称**
    import re
    
    # 尝试多种模式匹配
    patterns = [
        r'建议您挂\*\*([^*]+)\*\*',  # 建议您挂**科室名称**
        r'建议挂\*\*([^*]+)\*\*',    # 建议挂**科室名称**
        r'建议您立即前往\*\*([^*]+)\*\*',  # 建议您立即前往**科室名称**
        r'建议您挂号至?\*\*([^*]+)\*\*',  # 建议您挂号至**科室名称**
        r'推荐科室[：:]\s*\*\*([^*]+)\*\*',  # 推荐科室：**科室名称**
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            dept = match.group(1).strip()
            # 清理可能的后缀
            dept = dept.replace('门诊', '').strip()
            if dept:
                return dept
    
    # 如果没有匹配到，尝试查找所有可能的科室名称
    departments = [
        "中医科", "产科", "介入血管科", "儿科", "儿童保健",
        "全科医学科", "内分泌", "创面外科", "医学美容科", "口腔科",
        "呼吸与危重症医学科", "咳喘专病", "哮喘专病", "妇科", "康复科",
        "心血管内科", "感染性疾病科", "慢阻肺专病", "新生儿", "普通外科",
        "泌尿外科", "消化内科", "消化外科", "疼痛科", "皮肤科",
        "眼科", "神经内科", "神经外科", "精神心理科", "老年病科",
        "耳鼻咽喉头颈外科", "肛肠科", "肾脏内科", "肿瘤科、甲乳外科", "肿瘤科", "甲乳外科", "胸外科",
        "营养科", "血液透析中心", "针灸科", "骨外科", "麻醉科"
    ]
    
    for dept in departments:
        if dept in text:
            return dept
    
    return "未识别"


def normalize_department(dept: str) -> str:
    """标准化科室名称，用于比较"""
    # 移除"门诊"、"科"等后缀，只保留核心名称
    dept = dept.replace('门诊', '').replace('科室', '').strip()
    
    # 处理特殊情况
    if '骨外' in dept:
        return '骨外科'
    if '产科' in dept:
        return '产科'
    if '儿童保健' in dept:
        return '儿童保健'
    if '泌尿外科' in dept:
        return '泌尿外科'
    if '肿瘤' in dept or '甲乳' in dept:
        return '肿瘤科、甲乳外科'
    if '新生儿' in dept:
        return '新生儿'
    if '内分泌' in dept:
        return '内分泌'
    
    return dept


def evaluate_single_case(case_id: int, test_case: Dict, reasoner: Reasoner, max_new_tokens: int) -> Dict:
    """评估单个测试用例"""
    print(f"\n{'='*80}")
    print(f"测试用例 {case_id}")
    print(f"{'='*80}")
    
    input_text = test_case['input']
    expected_output = test_case['output']
    expected_dept = test_case.get('metadata', {}).get('department', '')
    
    print(f"患者咨询:\n{input_text}\n")
    print(f"预期科室: {expected_dept}")
    
    # 执行推理
    start_time = time.time()
    try:
        actual_output = reasoner.triage(user_text=input_text, max_new_tokens=max_new_tokens)
        inference_time = time.time() - start_time
        success = True
        error_msg = None
    except Exception as e:
        actual_output = ""
        inference_time = time.time() - start_time
        success = False
        error_msg = str(e)
        print(f" 推理失败: {error_msg}")
    
    # 提取实际输出的科室
    if success:
        actual_dept = extract_department(actual_output)
        print(f"预测科室: {actual_dept}")
        print(f"推理耗时: {inference_time:.2f}s")
        
        # 判断是否正确
        normalized_expected = normalize_department(expected_dept)
        normalized_actual = normalize_department(actual_dept)
        
        is_correct = normalized_expected == normalized_actual or normalized_expected in normalized_actual or normalized_actual in normalized_expected
        
        if is_correct:
            print("预测正确")
        else:
            print("预测错误")
            print(f"   标准化预期: {normalized_expected}")
            print(f"   标准化预测: {normalized_actual}")
        
        print("-" * 80)
        print(f"AI 完整输出:\n{actual_output.strip()}")
        print("=" * 80)
    else:
        actual_dept = "推理失败"
        is_correct = False
    
    return {
        "case_id": case_id,
        "input": input_text,
        "expected_department": expected_dept,
        "predicted_department": actual_dept,
        "expected_output": expected_output,
        "actual_output": actual_output,
        "is_correct": is_correct,
        "success": success,
        "error_msg": error_msg,
        "inference_time": inference_time
    }


def generate_summary_stats(results: List[Dict]) -> Dict:
    """生成统计摘要"""
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r['success'])
    correct_cases = sum(1 for r in results if r['is_correct'])
    
    total_time = sum(r['inference_time'] for r in results)
    avg_time = total_time / total_cases if total_cases > 0 else 0
    
    # 准确率基于所有测试用例（包括失败的）
    accuracy = correct_cases / total_cases if total_cases > 0 else 0
    success_rate = successful_cases / total_cases if total_cases > 0 else 0
    
    # 统计各科室的准确率（包括失败的情况）
    dept_stats = {}
    for result in results:
        expected_dept = normalize_department(result['expected_department'])
        if expected_dept not in dept_stats:
            dept_stats[expected_dept] = {'total': 0, 'correct': 0, 'failed': 0}
        dept_stats[expected_dept]['total'] += 1
        
        if not result['success']:
            dept_stats[expected_dept]['failed'] += 1
        elif result['is_correct']:
            dept_stats[expected_dept]['correct'] += 1
    
    # 计算每个科室的准确率
    dept_accuracy = {}
    for dept, stats in dept_stats.items():
        dept_accuracy[dept] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return {
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "correct_cases": correct_cases,
        "failed_cases": total_cases - successful_cases,
        "accuracy": accuracy,
        "success_rate": success_rate,
        "total_time": total_time,
        "avg_time_per_case": avg_time,
        "department_stats": dept_stats,
        "department_accuracy": dept_accuracy
    }


def save_results(results: List[Dict], summary: Dict, output_dir: str):
    """保存测试结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    detailed_file = output_path / f"detailed_results_{timestamp}.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("医疗分诊模型测试详细结果 (100题完整测试)\n")
        f.write("="*100 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试用例总数: {len(results)} 题\n")
        f.write("="*100 + "\n\n")
        
        for result in results:
            f.write(f"\n{'='*100}\n")
            f.write(f"测试用例 {result['case_id']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"患者咨询:\n{result['input']}\n\n")
            f.write(f"预期科室: {result['expected_department']}\n")
            f.write(f"预测科室: {result['predicted_department']}\n")
            f.write(f"推理状态: {'成功' if result['success'] else '失败'}\n")
            f.write(f"预测结果: {'正确' if result['is_correct'] else '❌ 错误'}\n")
            f.write(f"推理耗时: {result['inference_time']:.2f}s\n")
            
            if not result['success']:
                f.write(f"错误信息: {result['error_msg']}\n")
            
            f.write(f"\n{'-'*100}\n")
            f.write("AI 完整输出:\n")
            f.write(f"{result['actual_output']}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"预期输出:\n")
            f.write(f"{result['expected_output']}\n")
            f.write("="*100 + "\n")
    
    # 保存总结报告
    summary_file = output_path / f"summary_report_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("医疗分诊模型测试总结报告 (100题完整测试)\n")
        f.write("="*100 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【整体统计】\n")
        f.write(f"  - 测试用例总数: {summary['total_cases']} 题\n")
        f.write(f"  - 推理成功数: {summary['successful_cases']} 题\n")
        f.write(f"  - 推理失败数: {summary['failed_cases']} 题\n")
        f.write(f"  - 预测正确数: {summary['correct_cases']} 题\n")
        f.write(f"  - 推理成功率: {summary['success_rate']*100:.2f}%\n")
        f.write(f"  - 预测准确率: {summary['accuracy']*100:.2f}% (基于全部100题)\n")
        f.write(f"  - 总推理时间: {summary['total_time']:.2f}s\n")
        f.write(f"  - 平均推理时间: {summary['avg_time_per_case']:.2f}s/题\n\n")
        
        f.write("【各科室准确率统计】\n")
        sorted_depts = sorted(summary['department_accuracy'].items(), key=lambda x: x[1], reverse=True)
        for dept, acc in sorted_depts:
            stats = summary['department_stats'][dept]
            f.write(f"  - {dept:25s}: {acc*100:6.2f}%  ({stats['correct']}/{stats['total']})")
            if stats['failed'] > 0:
                f.write(f"  [推理失败: {stats['failed']}]")
            f.write("\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("【错误案例分析】\n")
        f.write("="*100 + "\n")
        
        error_count = 0
        for result in results:
            if not result['is_correct']:
                error_count += 1
                f.write(f"\n错误案例 {error_count} (题号: {result['case_id']})\n")
                f.write(f"患者咨询: {result['input']}\n")
                f.write(f"预期科室: {result['expected_department']}\n")
                f.write(f"预测科室: {result['predicted_department']}\n")
                if not result['success']:
                    f.write(f"失败原因: {result['error_msg']}\n")
                else:
                    # 显示标准化后的比较
                    norm_expected = normalize_department(result['expected_department'])
                    norm_actual = normalize_department(result['predicted_department'])
                    f.write(f"标准化预期: {norm_expected}\n")
                    f.write(f"标准化预测: {norm_actual}\n")
                f.write("-" * 100 + "\n")
        
        if error_count == 0:
            f.write("\n 恭喜！所有测试用例均预测正确！\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("测试完成\n")
        f.write("="*100 + "\n")
    
    # 保存JSON格式的结果
    json_file = output_path / f"results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存:")
    print(f"  - 详细结果: {detailed_file}")
    print(f"  - 总结报告: {summary_file}")
    print(f"  - JSON数据: {json_file}")


def main():
    """主函数"""
    # 配置参数
    MODEL_PATH = "Qwen/Qwen2.5-7B"
    ADAPTER_PATH = "/home/ma-user/work/output/qwen_triage/fp16/final/adapter_model"
    CACHE_DIR = "/home/ma-user/work/model"
    DEVICE = "Ascend"
    DEVICE_ID = int(os.environ.get("DEVICE_ID", 0))
    DTYPE = "fp16"
    MAX_NEW_TOKENS = 1024
    
    TEST_DATA_PATH = "/home/ma-user/work/dataset/test.txt"
    OUTPUT_DIR = "/home/ma-user/work/output/result"
    
    # 设置设备ID
    os.environ["DEVICE_ID"] = str(DEVICE_ID)
    
    print("="*100)
    print(" 医疗分诊模型完整测试 (100题)")
    print("="*100)
    print(f"[Info] MindSpore 2.7.0 / MindNLP 0.5.1 / Device {DEVICE}:{DEVICE_ID}")
    print(f"[Info] 测试数据: {TEST_DATA_PATH}")
    print(f"[Info] 输出目录: {OUTPUT_DIR}\n")
    
    # 加载测试数据
    print("[Step 1] 加载测试数据...")
    test_cases = load_test_data_from_txt(TEST_DATA_PATH)
    print(f"成功加载 {len(test_cases)} 题测试用例")
    print(f"   - 基础覆盖题: 40 题 (每个科室1题)")
    print(f"   - 常见症状拓展题: 60 题 (重点科室补充)\n")
    
    # 初始化 Reasoner
    print("[Step 2] 初始化推理模型...")
    print(f"  - Base Model:  {MODEL_PATH}")
    print(f"  - Adapter:     {ADAPTER_PATH}")
    print(f"  - Cache Dir:   {CACHE_DIR}")
    print(f"  - Device:      {DEVICE}:{DEVICE_ID}")
    print(f"  - Dtype:       {DTYPE}")
    
    start_time = time.time()
    reasoner = Reasoner(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        device=DEVICE,
        dtype=DTYPE,
        cache_dir=CACHE_DIR,
        device_id=DEVICE_ID,
    )
    init_time = time.time() - start_time
    print(f"模型初始化完成 (耗时 {init_time:.2f}s)\n")
    
    # 执行测试
    print("[Step 3] 开始批量测试...")
    print(f"共 {len(test_cases)} 题，预计耗时较长，请耐心等待...\n")
    
    results = []
    for idx, test_case in enumerate(test_cases, start=1):
        result = evaluate_single_case(idx, test_case, reasoner, MAX_NEW_TOKENS)
        results.append(result)
    
    # 生成统计摘要
    print("\n[Step 4] 生成统计报告...")
    summary = generate_summary_stats(results)
    
    # 保存结果
    print("\n[Step 5] 保存测试结果...")
    save_results(results, summary, OUTPUT_DIR)
    
    # 打印总结
    print("\n" + "="*100)
    print("【测试总结】")
    print("="*100)
    print(f"测试用例总数: {summary['total_cases']} 题")
    print(f"推理成功数:   {summary['successful_cases']} 题")
    print(f"推理失败数:   {summary['failed_cases']} 题")
    print(f"预测正确数:   {summary['correct_cases']} 题")
    print(f"推理成功率:   {summary['success_rate']*100:.2f}%")
    print(f"预测准确率:   {summary['accuracy']*100:.2f}% (基于全部100题)")
    print(f"总推理时间:   {summary['total_time']:.2f}s")
    print(f"平均推理时间: {summary['avg_time_per_case']:.2f}s/题")
    print("="*100)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

