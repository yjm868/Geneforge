import flet as ft
import threading
import os
import time
import json
import cv2
import base64
from openai import OpenAI

# ================================================================
# ⚠️ 请填写你的阿里云 DashScope API Key
# 去 https://dashscope.aliyun.com/ 用手机号注册登录 -> 创建 Key
# =====================================================
import os
# 如果没安装 python-dotenv，记得 pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()

ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY")
# --- 视觉大脑与文本大脑 ---
client_vision = OpenAI(
    api_key=ALIBABA_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
client_text = OpenAI(
    api_key=ALIBABA_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 导入依赖 ---
try:
    import chromadb
    from chromadb.utils import embedding_functions
    chroma_client = chromadb.PersistentClient(path="./chroma_memory_db")
    collection = chroma_client.get_or_create_collection(
        name="long_term_memory",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
    )
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

# 全局状态
current_goal = ""
current_memory = ""
video_learning_running = False
camera_running = False

# ================= 核心文本互攻逻辑 =================
def call_llm(prompt):
    try:
        response = client_text.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 文本推理 API 出错: {e}")
        return None

def agent_a_suggestion(goal, memory):
    return call_llm(f"你是激进派，目标：{goal}。当前进度摘要：{memory}。请提出一步大胆的方案。")

def agent_b_attack(goal, memory, a_say):
    return call_llm(f"你是找茬专家，目标：{goal}。当前进度摘要：{memory}。激进派方案说：{a_say}。请找出3个漏洞并给出改进。")

def leader_decide(goal, memory, a_say, b_say):
    return call_llm(f"你是最高决策者，目标：{goal}。当前进度摘要：{memory}。A说：{a_say}。B反驳：{b_say}。请结合双方智慧融合成最佳执行方案。")

# ================= 视觉识别 =================
def call_vision_llm(frame, prompt_text):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        response = client_vision.chat.completions.create(
            model="qwen-vl-plus",  
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ 视觉识别接口报错：{str(e)}"

# ================= 视频学习 =================
def start_video_learning(video_path, log_area, page):
    global video_learning_running
    if video_learning_running:
        log_area.controls.append(ft.Text("⚠️ 当前已有视频在学习中...", color="red"))
        page.update()
        return
    if not HAS_MOVIEPY:
        log_area.controls.append(ft.Text("⚠️ 缺失 moviepy 库，无法处理视频。", color="red"))
        page.update()
        return

    def learn_process():
        global video_learning_running
        video_learning_running = True
        log_area.controls.append(ft.Text(f"📹 开始用视觉大脑学习视频...", color="green"))
        page.update()
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            frame_interval = max(1, int(duration / 5)) if duration < 15 else 15
            curr_time = 0
            success_count = 0
            while curr_time < duration:
                frame = clip.get_frame(curr_time)
                frame = (frame * 255).astype('uint8')
                vision_result = call_vision_llm(
                    frame, 
                    f"这是一段视频第 {int(curr_time)} 秒的画面。请用简洁、客观、准确的 20-30 字描述画面里的主要自然景象、动植物状态、或周边环境。"
                )
                if vision_result:
                    print(f"🔍 阿里云视觉识别结果：{vision_result}")
                    log_area.controls.append(ft.Text(f"👁️ 识别结果：{vision_result}", color="cyan"))
                    page.update()
                    
                    if HAS_CHROMA and "报错" not in vision_result:
                        collection.add(
                            documents=[vision_result], 
                            metadatas={"source": os.path.basename(video_path), "timestamp": int(curr_time)}, 
                            ids=[f"{video_path}_{int(curr_time)}"]
                        )
                        success_count += 1
                curr_time += frame_interval
            clip.close()
            log_area.controls.append(ft.Text(f"✅ 视觉学习完成！存入 {success_count} 条真实记忆。", color="green"))
            page.update()
        except Exception as err:
            log_area.controls.append(ft.Text(f"❌ 视频学习失败：{err}", color="red"))
            page.update()
        finally:
            video_learning_running = False

    threading.Thread(target=learn_process).start()

# ================= 环境感知 =================
def sense_environment(log_area, page):
    try:
        log_area.controls.append(ft.Text("📷🎙️ 环境感知中...", color="purple"))
        page.update()
        seen_text = "无画面"
        # 第一处修改：这里加上 cv2.CAP_DSHOW
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        cap.release()
        if ret:
            seen_desc = call_llm("描述这张抓拍画面的核心物体或场景，20字以内。")
            seen_text = f"画面内容：{seen_desc}"
        full_sense = f"看到：{seen_text}"
        log_area.controls.append(ft.Text(f"🌍 感知结果：\n{full_sense}", color="cyan"))
        if HAS_CHROMA:
            collection.add(documents=[full_sense], metadatas={"time": time.strftime("%Y-%m-%d %H:%M:%S")}, ids=[str(time.time())])
        page.update()
    except Exception as err:
        log_area.controls.append(ft.Text(f"❌ 感知过程异常：{err}", color="red"))
        page.update()

# ================= 实时摄像头视频对话 =================
def start_realtime_video(log_area, page):
    global camera_running
    if camera_running:
        log_area.controls.append(ft.Text("⚠️ 摄像头已经在运行中", color="red"))
        page.update()
        return

    def video_loop():
        global camera_running
        camera_running = True
        log_area.controls.append(ft.Text("📷 正在打开摄像头...", color="purple"))
        page.update()
        
        # 第二处修改：这里也加上 cv2.CAP_DSHOW
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log_area.controls.append(ft.Text("❌ 无法打开摄像头，请检查权限或摄像头是否被占用", color="red"))
            camera_running = False
            page.update()
            return

        while camera_running:
            ret, frame = cap.read()
            if ret:
                vision_desc = call_vision_llm(frame, "描述摄像头实时画面中的主要场景，20字以内。")
                if vision_desc and "报错" not in vision_desc:
                    log_area.controls.append(ft.Text(f"👁️ 实时画面：{vision_desc}", color="cyan"))
                    a_say = agent_a_suggestion(f"根据实时画面：{vision_desc}", "")
                    b_say = agent_b_attack(f"根据实时画面：{vision_desc}", "", a_say)
                    final = leader_decide(f"根据实时画面：{vision_desc}", "", a_say, b_say)
                    log_area.controls.append(ft.Text(f"👑 高层实时决策：{final}", color="cyan", weight="bold"))
                    page.update()
                    time.sleep(2) # 每2秒分析一次
            else:
                log_area.controls.append(ft.Text("⚠️ 摄像头断开连接", color="red"))
                camera_running = False
                page.update()
                break
        cap.release()
        log_area.controls.append(ft.Text("📷 摄像头已关闭", color="grey"))
        page.update()

    def stop_video():
        global camera_running
        camera_running = False

    threading.Thread(target=video_loop).start()
    # 提供一个停止按钮的逻辑（放在UI里）
    # 这里简单处理，开始运行后，再次点击即可停止（逻辑需看实际情况，这里简单做一次触发）
    # 为方便演示，如果点击了开始，之后只能关掉程序停止，你可以加一个全局标志来控制退出
    # 本次代码暂不实现“停止”按钮，需要关掉整个窗口或重启程序停止。

# ================= 桌面 GUI 界面 =================
def main(page: ft.Page):
    page.title = "🤖 塔式多智能体引擎 (摄像头修复版)"
    page.window.width = 1000
    page.window.height = 750
    page.theme_mode = ft.ThemeMode.DARK
    
    header = ft.Text("多智能体塔式互攻系统 (实时视频推演)", size=24, weight=ft.FontWeight.BOLD, color="cyan")
    goal_input = ft.TextField(label="输入当前执行目标", width=450, bgcolor="black12")
    log_area = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)

    def start_attack(e):
        global current_goal, current_memory
        if not goal_input.value:
            log_area.controls.append(ft.Text("⚠️ 请先输入任务目标！", color="red"))
            page.update()
            return
        current_goal = goal_input.value
        log_area.controls.append(ft.Text(f"🎯 初始目标设定为: {current_goal}", color="green"))
        current_memory = "任务初始状态。"
        page.update()

    def run_ai_cycle(e):
        def cycle():
            global current_goal, current_memory
            if current_goal == "":
                log_area.controls.append(ft.Text("⚠️ 请先点击「初始化目标」再攻击！", color="red"))
                page.update()
                return
            
            memory_context = ""
            if HAS_CHROMA:
                try:
                    results = collection.query(query_texts=[current_goal], n_results=3)
                    if results['documents'] and results['documents'][0]:
                        memory_context = "【系统已检索到相关视觉记忆】\n" + "\n".join(results['documents'][0])
                        log_area.controls.append(ft.Text(f"📚 已自动加载视觉记忆辅助推理。", color="cyan"))
                        page.update()
                except Exception:
                    pass
            
            final_goal = f"{current_goal}\n\n{memory_context}"
            
            log_area.controls.append(ft.Divider(height=10, color="grey"))
            log_area.controls.append(ft.Text(f"🔽 发动底层互攻循环...", color="yellow"))
            page.update()
            
            a_plan = agent_a_suggestion(final_goal, current_memory)
            log_area.controls.append(ft.Text(f"🧠 Agent A: {a_plan}", color="white"))
            page.update()
            
            b_attack = agent_b_attack(final_goal, current_memory, a_plan)
            log_area.controls.append(ft.Text(f"💢 Agent B: {b_attack}", color="orange"))
            page.update()
            
            final = leader_decide(final_goal, current_memory, a_plan, b_attack)
            log_area.controls.append(ft.Text(f"👑 高层决策: {final}", color="cyan", weight="bold"))
            page.update()
            
            current_memory = call_llm(f"目标：{current_goal}。做了：{final}。请用20字总结客观进度。")
            log_area.controls.append(ft.Text(f"📋 当前进度: {current_memory}", color="green"))
            page.update()
        threading.Thread(target=cycle).start()

    # --- 按钮逻辑 ---
    def on_file_picked(e):
        if e.files:
            video_path = e.files[0].path
            log_area.controls.append(ft.Text(f"📹 已选择文件: {os.path.basename(video_path)}", color="green"))
            page.update()
            start_video_learning(video_path, log_area, page)
        else:
            log_area.controls.append(ft.Text("⏹️ 用户取消了文件选择。", color="grey"))
            page.update()

    file_picker = ft.FilePicker()
    file_picker.on_result = on_file_picked
    page.overlay.append(file_picker)

    def open_native_file_dialog(e):
        file_picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "avi", "mov", "mkv"]
        )

    def sense_env_btn_click(e):
        threading.Thread(target=sense_environment, args=(log_area, page)).start()

    def realtime_video_btn_click(e):
        start_realtime_video(log_area, page)

    btn_init = ft.FilledButton("🎯 初始化目标", on_click=start_attack, style=ft.ButtonStyle(color="white", bgcolor="blue"))
    btn_attack = ft.FilledButton("⚔️ 发动一轮互攻", on_click=run_ai_cycle, style=ft.ButtonStyle(color="white", bgcolor="blue"))
    btn_learn = ft.FilledButton("📹 导入视频闲时学习", on_click=open_native_file_dialog, style=ft.ButtonStyle(color="white", bgcolor="green"))
    btn_sense = ft.FilledButton("📷🎙️ 环境感知", on_click=sense_env_btn_click, style=ft.ButtonStyle(color="white", bgcolor="orange"))
    btn_video = ft.FilledButton("📷 开始实时视频对话", on_click=realtime_video_btn_click, style=ft.ButtonStyle(color="white", bgcolor="purple"))

    page.add(
        header,
        ft.Row([goal_input, btn_init, btn_attack, btn_sense, btn_learn, btn_video], wrap=True, alignment=ft.MainAxisAlignment.START),
        ft.Container(content=log_area, expand=True, bgcolor="black87", border_radius=10, padding=10)
    )

if __name__ == "__main__":
    ft.app(target=main)