import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
from reasoning import Reasoner

# ================= 配置区域 =================
MODEL_PATH = "Qwen/Qwen2.5-7B"
ADAPTER_PATH = "/home/ma-user/work/output/qwen_triage/fp16/final/adapter_model"
CACHE_DIR = "/home/ma-user/work/model"
DEVICE = "Ascend"
DTYPE = "fp16"

# 全局模型实例
global_reasoner = None

def load_model_at_startup():
    """启动时自动加载模型"""
    global global_reasoner
    print("\n" + "="*50)
    print("正在初始化系统，请稍候...")
    print(f"基座模型: {MODEL_PATH}")
    print(f"适配器:   {ADAPTER_PATH}")
    print("="*50 + "\n")
    
    try:
        global_reasoner = Reasoner(
            model_path=MODEL_PATH,
            adapter_path=ADAPTER_PATH,
            device=DEVICE,
            dtype=DTYPE,
            cache_dir=CACHE_DIR,
        )
        return True
    except Exception as e:
        print(f"\n严重错误: 模型加载失败!\n{str(e)}")
        return False

def predict(history, text):
    """
    推理函数
    """
    max_new_tokens = 512
    if not global_reasoner:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": "错误: 模型未加载，请检查后台日志。"})
        yield history
        return

    if not text.strip():
        yield history
        return

    # 1. 添加用户消息
    history.append({"role": "user", "content": text})
    yield history
    
    try:
        # 2. 初始化助手回复（流式）
        history.append({"role": "assistant", "content": ""})
        partial_response = ""
        
        # 3. 调用流式分诊方法
        for new_token in global_reasoner.stream_triage(user_text=text, max_new_tokens=max_new_tokens):
            partial_response += new_token
            history[-1]["content"] = partial_response
            yield history

    except Exception as e:
        import traceback
        traceback.print_exc()
        history.append({"role": "assistant", "content": f"生成出错: {str(e)}"})
        yield history

# ================= 界面构建 =================
def build_gemini_style_app():
    
    # 自定义 CSS (通过 Markdown 注入，兼容所有版本)
    custom_css = """
    <style>
    /* 全局字体和背景 */
    .gradio-container {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        background-color: #f7f9fb !important;
    }
    
    /* 标题样式 */
    #header-area {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e1e4e8;
    }
    #header-area h2 {
        color: #2c3e50;
        font-size: 28px !important;
        margin-bottom: 10px;
        font-weight: 600;
    }
    #header-area p {
        color: #7f8c8d;
        font-size: 16px;
    }

    /* 聊天窗口样式 */
    #chatbot {
        border: 1px solid #e1e4e8 !important;
        background: white !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    
    /* 调整聊天气泡（清淡医疗/马卡龙风格） */
    .message.user {
        background-color: #E0F2F1 !important;  /* 极淡薄荷绿 */
        color: #263238 !important;             /* 深蓝灰文字 */
        border-radius: 15px 15px 0 15px !important;
        border: 1px solid #B2DFDB !important;
    }
    .message.bot {
        background-color: #FAFAFA !important;  /* 纯净白灰 */
        color: #37474F !important;
        border-radius: 15px 15px 15px 0 !important;
        border: 1px solid #EEEEEE !important;
    }
    
    /* 按钮样式（柔和青色） */
    #send-btn {
        background: #80CBC4 !important;        /* 马卡龙青 */
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(77,182,172,0.2) !important;
    }
    #send-btn:hover {
        background: #4DB6AC !important;        /* 稍微加深 */
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(77,182,172,0.3) !important;
    }
    </style>
    """
    
    with gr.Blocks(title="AI 智能分诊助手") as demo:
        gr.HTML(custom_css)  # 注入 CSS
        
        with gr.Column(elem_id="main-container"):
            # 顶部标题区
            with gr.Column(elem_id="header-area"):
                gr.Markdown("## AI 智能分诊助手")
                gr.Markdown("您的专属医疗导诊顾问 | 24小时在线 | 智能匹配科室")
            
            # 对话框
            chatbot = gr.Chatbot(
                label="对话历史",
                elem_id="chatbot",
                height=600
            )
            
            # 输入区域
            with gr.Row(elem_classes="input-row"):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="请描述您的症状（例如：头痛、发烧、腹痛位置等）...",
                    scale=8,
                    container=False
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=80, elem_id="send-btn")
                clear_btn = gr.Button("清空", size="sm", variant="secondary", scale=1, min_width=80)
            
            # 底部提示 (原高级设置移除，保留提示)
            gr.Markdown("如遇紧急情况请直接拨打 120。", elem_id="footer-note")

        # 初始欢迎语 - 手机预约挂号场景
        def init_history():
            return [{"role": "assistant", "content": "您好！我是医院手机挂号系统的智能分诊助手。\n\n请详细描述您的症状（如：**发烧39度持续3天**、**右下腹剧痛**等），我会根据本院 **40 个门诊科室**为您推荐最合适的预约科室。\n\n**温馨提示**：如遇大出血、昏迷等危急情况，请立即拨打 **120** 或前往急诊。"}]

        # 事件绑定
        demo.load(init_history, outputs=[chatbot])
        
        msg.submit(predict, inputs=[chatbot, msg], outputs=[chatbot]) \
           .then(lambda: "", outputs=[msg]) 
           
        submit_btn.click(predict, inputs=[chatbot, msg], outputs=[chatbot]) \
                  .then(lambda: "", outputs=[msg])
        
        clear_btn.click(init_history, outputs=[chatbot])

    return demo

if __name__ == "__main__":
    # 1. 先加载模型
    success = load_model_at_startup()
    
    if success:
        # 2. 启动界面
        print("系统准备就绪，正在启动 Web 界面...")
        app = build_gemini_style_app()
        app.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        print("启动中止：模型加载失败。")
