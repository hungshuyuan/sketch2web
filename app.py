"""
手繪草圖 → 網頁（Sketch to Website）
本地執行：先啟動 ~/start_llm.sh，再 python app.py
Docker：docker compose up -d
"""
import os
import re
import io
import base64
import time
import html

import gradio as gr
from PIL import Image
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LAB_API_BASE", "http://127.0.0.1:8080/v1"),
    api_key=os.environ.get("LAB_API_KEY", "local"),
    timeout=600,  # 這個模型一定會先思考，回應可能要等一段時間
)
MODEL = os.environ.get("LAB_MODEL", "Qwen3.8-Flash-Next")

SYSTEM_PROMPT = """你是資深前端工程師。使用者會給你一張手繪的網頁草圖（wireframe）。
請把草圖轉成一個完整、可直接用瀏覽器開啟的單一 HTML 檔：
- 只用 HTML + Tailwind CSS（<script src="https://cdn.tailwindcss.com"></script>）與少量原生 JavaScript
- 忠實還原草圖中各區塊的相對位置、大小與層級（導覽列、標題、按鈕、卡片、表單、頁尾等）
- 草圖上的文字（包含打字輸入的文字方塊）照用；看不清楚的用合理的繁體中文占位文字
- 圖片用 https://placehold.co/寬x高 當佔位圖
- 需支援手機版（響應式）
- 只輸出一個 ```html 程式碼區塊，不要任何其他說明"""

OUT_PATH = "/tmp/generated.html"


# ---------------- 模型呼叫 ----------------

def to_data_url(img: Image.Image) -> str:
    """把圖片轉成白底 JPEG 並縮小，再編成 base64 data URL"""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")  # 透明背景補白，避免變黑底
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_html(text: str) -> str:
    """去掉思考內容，取出 ```html 區塊"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def render(code: str):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
    iframe = (
        f'<iframe srcdoc="{html.escape(code)}" '
        'style="width:100%;height:720px;border:1px solid #ddd;'
        'border-radius:8px;background:white"></iframe>'
    )
    return code, iframe, OUT_PATH


def partial_code(text: str) -> str:
    """串流中：取出目前已寫出的 html（結尾的 ``` 還沒出現也可以）"""
    m = re.search(r"```(?:html)?\s*(.*?)(```|$)", text, flags=re.S)
    return m.group(1) if m else ""


def split_think(content: str, reasoning: str):
    """有些伺服器把思考放在 content 的 <think> 裡，把它分出來"""
    m = re.match(r"\s*<think>(.*?)(</think>|$)(.*)", content, flags=re.S)
    if m:
        return reasoning + m.group(1), m.group(3)
    return reasoning, content


def stream_model(messages):
    """逐段讀取模型輸出，回傳 (思考內容, 正文, 是否結束)"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=16384,
        stream=True,
    )
    reasoning, content, last = "", "", 0.0
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        extra_fields = getattr(d, "model_extra", None) or {}
        r = (getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
             or extra_fields.get("reasoning_content") or extra_fields.get("reasoning"))
        if r:
            reasoning += r
        if d.content:
            content += d.content
        now = time.time()
        if now - last > 0.15:  # 每 0.15 秒更新一次畫面，避免瀏覽器太忙
            last = now
            yield reasoning, content, False
    yield reasoning, content, True


LIVE_FRAME = (
    '<div id="live-holder" style="height:720px">'
    '<iframe id="live-frame" '
    'style="width:100%;height:720px;border:1px solid #ddd;'
    'border-radius:8px;background:white"></iframe></div>'
)


def live_value(run_id, state, code):
    """傳給瀏覽器的格式：第一行是「執行編號|狀態」，之後是目前的程式碼"""
    return f"{run_id}|{state}\n{code}"


def run_stream(messages):
    """把模型輸出即時推到介面：預覽視窗邊寫邊畫，另外顯示思考過程與程式碼"""
    t0 = time.time()
    run_id = str(int(t0 * 1000))
    # 開始時：預覽換成空白的即時視窗，並停在「預覽」分頁
    yield ("連線中…", "", "", gr.update(), LIVE_FRAME, gr.update(),
           gr.Tabs(selected="preview"), live_value(run_id, "run", ""))
    reasoning, body = "", ""
    for raw_reasoning, content, done in stream_model(messages):
        reasoning, body = split_think(content, raw_reasoning)
        code = partial_code(body)
        secs = int(time.time() - t0)
        if code:
            status = f"正在繪製網頁…　{secs} 秒　已產生 {code.count(chr(10)) + 1} 行"
        else:
            status = f"模型思考中…　{secs} 秒　已思考 {len(reasoning)} 字"
        if not done:
            yield (status, reasoning, code, gr.update(), gr.update(), gr.update(),
                   gr.update(), live_value(run_id, "run", code))

    final = extract_html(body)
    if not final:
        raise gr.Error("模型沒有回傳程式碼，請再試一次")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(final)
    secs = int(time.time() - t0)
    # 完成：預覽視窗已經畫好了，不再重新載入，只通知瀏覽器收尾
    yield (f"完成　共 {secs} 秒　{final.count(chr(10)) + 1} 行", reasoning, final,
           final, gr.update(), OUT_PATH, gr.update(), live_value(run_id, "done", final))


def generate(data_url, extra):
    if not data_url or "," not in data_url:
        raise gr.Error("請先在畫布上畫草圖，或上傳底圖")
    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    user_text = "請把這張手繪草圖轉成網頁。"
    if extra:
        user_text += f"\n額外要求：{extra}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": to_data_url(img)}},
            {"type": "text", "text": user_text},
        ]},
    ]
    yield from run_stream(messages)


def refine(code, instruction):
    if not code:
        raise gr.Error("請先生成網頁")
    if not instruction:
        raise gr.Error("請輸入修改指令")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"這是目前的網頁：\n```html\n{code}\n```\n請依照以下要求修改，輸出完整的新版本：{instruction}"},
    ]
    yield from run_stream(messages)


# ---------------- 圖示（Lucide 風格線條 SVG） ----------------

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
os.makedirs(ICON_DIR, exist_ok=True)

ICON_PATHS = {
    "pencil": '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    "maximize": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "pointer": '<path d="M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z"/>',
    "square": '<rect width="18" height="18" x="3" y="3" rx="2"/>',
    "type": '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
}


def svg(name, color="currentColor", size=24):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{ICON_PATHS[name]}</svg>'
    )


def icon_file(name, color):
    """Gradio 按鈕圖示要用檔案；主要按鈕用白色，其他用中灰色（深淺色主題都看得到）"""
    path = os.path.join(ICON_DIR, f"{name}.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg(name, color))
    return path


# ---------------- 畫布（Fabric.js） ----------------

FABRIC_HEAD = '<script src="https://cdn.jsdelivr.net/npm/fabric@5.5.2/dist/fabric.min.js"></script>'

CANVAS_HTML = """
<div id="sk-wrap">
  <div id="sk-toolbar">
    <button class="sk-btn active" data-tool="select" title="選取、移動、縮放物件">{pointer}<span>選取</span></button>
    <button class="sk-btn" data-tool="pen" title="自由繪製">{pencil}<span>畫筆</span></button>
    <button class="sk-btn" data-tool="rect" title="拖曳畫出方框">{square}<span>方框</span></button>
    <button class="sk-btn" data-tool="text" title="點畫布任一處新增文字方塊">{type}<span>文字</span></button>
    <span class="sk-sep"></span>
    <select id="sk-size" title="線條粗細／文字大小">
      <option value="s">細／小字</option>
      <option value="m" selected>中／中字</option>
      <option value="l">粗／大字</option>
    </select>
    <span class="sk-sep"></span>
    <button class="sk-btn" id="sk-undo" title="復原（Ctrl+Z）">{undo}<span>復原</span></button>
    <button class="sk-btn" id="sk-delete" title="刪除選取的物件（Delete）">{x}<span>刪除選取</span></button>
    <button class="sk-btn" id="sk-clear" title="清空整個畫布">{trash}<span>清空</span></button>
    <label class="sk-btn" for="sk-file" title="上傳手繪草圖照片當底圖">{upload}<span>上傳底圖</span></label>
    <input type="file" id="sk-file" accept="image/*" hidden>
  </div>
  <div id="sk-stage"><canvas id="sk-canvas"></canvas></div>
  <div class="sk-hint">文字方塊：選「文字」後點畫布即可輸入，雙擊已有的文字可再編輯，用「選取」可拖曳位置。也可以直接 Ctrl+V 貼上圖片當底圖。</div>
</div>
"""
for _name in ["pointer", "pencil", "square", "type", "undo", "x", "trash", "upload"]:
    CANVAS_HTML = CANVAS_HTML.replace("{" + _name + "}", svg(_name, size=18))

CANVAS_JS = """
() => {
  const W = 1600, H = 1000;  // 匯出解析度
  const SIZES = {
    s: { pen: 3, stroke: 2, font: 28 },
    m: { pen: 5, stroke: 4, font: 44 },
    l: { pen: 10, stroke: 7, font: 64 },
  };
  const INK = '#111111';

  function start() {
    const el = document.getElementById('sk-canvas');
    if (!window.fabric || !el) { setTimeout(start, 200); return; }
    if (window.skCanvas) return;

    const c = new fabric.Canvas('sk-canvas', { backgroundColor: '#ffffff', preserveObjectStacking: true });
    window.skCanvas = c;
    let tool = 'select', size = 'm', zoom = 1;
    let history = [], restoring = false, rect = null, origin = null;

    // ---- 歷史紀錄（復原） ----
    function save() {
      if (restoring) return;
      history.push(JSON.stringify(c.toJSON()));
      if (history.length > 60) history.shift();
    }
    function undo() {
      if (history.length < 2) return;
      history.pop();
      restoring = true;
      c.loadFromJSON(JSON.parse(history[history.length - 1]), () => {
        restoring = false;
        setTool(tool);
        c.requestRenderAll();
      });
    }
    c.on('object:added', save);
    c.on('object:modified', save);
    c.on('text:editing:exited', save);

    // ---- 依容器寬度縮放顯示，邏輯座標固定 1600x1000 ----
    const stage = document.getElementById('sk-stage');
    function resize() {
      const avail = stage.clientWidth || 600;
      const maxH = window.innerHeight * 0.78;
      const w = Math.floor(Math.min(avail, maxH * W / H));
      if (Math.abs(w / W - zoom) < 0.001 && c.getWidth() === w) return;
      zoom = w / W;
      c.setDimensions({ width: w, height: Math.round(w * H / W) });
      c.setZoom(zoom);
      c.requestRenderAll();
    }
    new ResizeObserver(() => requestAnimationFrame(resize)).observe(stage);
    window.addEventListener('resize', resize);

    // ---- 工具切換 ----
    function setTool(t) {
      tool = t;
      c.isDrawingMode = (t === 'pen');
      if (c.isDrawingMode) {
        c.freeDrawingBrush = new fabric.PencilBrush(c);
        c.freeDrawingBrush.color = INK;
        c.freeDrawingBrush.width = SIZES[size].pen;
      }
      c.selection = (t === 'select');
      c.defaultCursor = (t === 'select') ? 'default' : (t === 'text' ? 'text' : 'crosshair');
      c.forEachObject(o => {
        o.selectable = (t === 'select');
        o.evented = (t === 'select') || (t === 'text' && o.type === 'i-text');
      });
      if (t !== 'select') c.discardActiveObject();
      c.requestRenderAll();
      document.querySelectorAll('#sk-toolbar [data-tool]').forEach(b =>
        b.classList.toggle('active', b.dataset.tool === t));
    }
    document.querySelectorAll('#sk-toolbar [data-tool]').forEach(b =>
      b.addEventListener('click', () => setTool(b.dataset.tool)));

    // ---- 方框與文字方塊 ----
    c.on('mouse:down', opt => {
      const p = c.getPointer(opt.e);
      if (tool === 'rect') {
        origin = p;
        rect = new fabric.Rect({
          left: p.x, top: p.y, width: 1, height: 1, fill: 'transparent',
          stroke: INK, strokeWidth: SIZES[size].stroke, strokeUniform: true,
          selectable: false, evented: false,
        });
        restoring = true; c.add(rect); restoring = false;
      } else if (tool === 'text') {
        if (opt.target && opt.target.type === 'i-text') {
          const t = opt.target;
          setTool('select');
          c.setActiveObject(t);
          t.enterEditing();
          return;
        }
        const t = new fabric.IText('文字', {
          left: p.x, top: p.y, fontSize: SIZES[size].font, fill: INK,
          fontFamily: '"Noto Sans TC", "Microsoft JhengHei", "PingFang TC", sans-serif',
        });
        c.add(t);
        setTool('select');
        c.setActiveObject(t);
        t.enterEditing();
        t.selectAll();
        c.requestRenderAll();
      }
    });
    c.on('mouse:move', opt => {
      if (!rect) return;
      const p = c.getPointer(opt.e);
      rect.set({
        left: Math.min(p.x, origin.x), top: Math.min(p.y, origin.y),
        width: Math.abs(p.x - origin.x), height: Math.abs(p.y - origin.y),
      });
      c.requestRenderAll();
    });
    c.on('mouse:up', () => {
      if (!rect) return;
      if (rect.width < 6 && rect.height < 6) {
        restoring = true; c.remove(rect); restoring = false;
      } else {
        rect.setCoords();
        save();
      }
      rect = null;
    });
    c.on('path:created', e => { e.path.selectable = false; e.path.evented = false; });

    // ---- 粗細／字級 ----
    document.getElementById('sk-size').addEventListener('change', e => {
      size = e.target.value;
      if (c.isDrawingMode) c.freeDrawingBrush.width = SIZES[size].pen;
      const a = c.getActiveObject();
      if (a && a.type === 'i-text') { a.set('fontSize', SIZES[size].font); c.requestRenderAll(); save(); }
      else if (a && a.type === 'rect') { a.set('strokeWidth', SIZES[size].stroke); c.requestRenderAll(); save(); }
    });

    // ---- 刪除、清空、復原 ----
    function deleteSelected() {
      const objs = c.getActiveObjects();
      if (!objs.length) return;
      restoring = true;
      objs.forEach(o => c.remove(o));
      restoring = false;
      c.discardActiveObject();
      c.requestRenderAll();
      save();
    }
    function clearAll() {
      restoring = true;
      c.clear();
      c.backgroundColor = '#ffffff';
      restoring = false;
      c.requestRenderAll();
      save();
    }
    document.getElementById('sk-undo').addEventListener('click', undo);
    document.getElementById('sk-delete').addEventListener('click', deleteSelected);
    document.getElementById('sk-clear').addEventListener('click', clearAll);

    function typingInField() {
      const a = document.activeElement;
      const editingText = c.getActiveObject() && c.getActiveObject().isEditing;
      return editingText || (a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.isContentEditable));
    }
    document.addEventListener('keydown', e => {
      if (typingInField()) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (c.getActiveObjects().length) { deleteSelected(); e.preventDefault(); }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        undo(); e.preventDefault();
      }
    });

    // ---- 底圖：上傳或貼上 ----
    function setBackground(url) {
      fabric.Image.fromURL(url, img => {
        const s = Math.min(W / img.width, H / img.height);
        c.setBackgroundImage(img, () => { c.requestRenderAll(); save(); }, {
          scaleX: s, scaleY: s, originX: 'left', originY: 'top',
          left: (W - img.width * s) / 2, top: (H - img.height * s) / 2,
        });
      });
    }
    function readImage(file) {
      const r = new FileReader();
      r.onload = () => setBackground(r.result);
      r.readAsDataURL(file);
    }
    document.getElementById('sk-file').addEventListener('change', e => {
      if (e.target.files[0]) readImage(e.target.files[0]);
      e.target.value = '';
    });
    document.addEventListener('paste', e => {
      if (typingInField()) return;
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const it of items) {
        if (it.type.startsWith('image/')) { readImage(it.getAsFile()); e.preventDefault(); break; }
      }
    });

    // ---- 匯出給後端 ----
    window.skExport = () => {
      if (!c.getObjects().length && !c.backgroundImage) return '';
      c.discardActiveObject();
      c.renderAll();
      return c.toDataURL({ format: 'jpeg', quality: 0.92, multiplier: 1 / zoom });
    };

    resize();
    setTool('select');
    save();
  }
  start();
}
"""

EXPORT_JS = "(d) => [window.skExport ? window.skExport() : '']"

# 把預覽 iframe 切成全螢幕（按 Esc 離開）
LIVE_JS = """
() => {
  // 每 120ms 檢查一次後端送來的程式碼，只把「新增的部分」寫進預覽視窗，
  // 瀏覽器會自己把寫到一半的 HTML 補完並顯示，所以網頁會邊寫邊長出來。
  let runId = null, written = '', closed = false, zoomed = false;

  // ---------- 放大到整個視窗（不需要使用者點擊，所以不用瀏覽器原生全螢幕） ----------
  const EXIT_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/></svg>';

  function ui() {
    let bar = document.getElementById('zoom-bar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'zoom-bar';
      bar.innerHTML = '<span id="zoom-status"></span><button id="zoom-exit" title="離開全螢幕（Esc）">' + EXIT_ICON + '<span>離開全螢幕</span></button>';
      document.body.appendChild(bar);
      document.getElementById('zoom-exit').addEventListener('click', exitZoom);
    }
    return bar;
  }

  function autoZoomOn() {
    const cb = document.querySelector('#auto-fs input[type=checkbox]');
    return !cb || cb.checked;
  }

  function enterZoom() {
    const f = document.getElementById('live-frame');
    if (!f || zoomed) return;
    zoomed = true;
    const r = f.getBoundingClientRect();
    // 先固定在原本的位置與大小，下一格畫面再放大到整個視窗（產生放大動畫）
    Object.assign(f.style, {
      position: 'fixed', zIndex: 2147483000, margin: 0, transition: 'none',
      top: r.top + 'px', left: r.left + 'px', width: r.width + 'px', height: r.height + 'px',
      boxShadow: '0 20px 60px rgba(0,0,0,.35)',
    });
    document.documentElement.style.overflow = 'hidden';
    void f.offsetWidth;
    Object.assign(f.style, {
      transition: 'top .55s cubic-bezier(.2,.8,.2,1), left .55s cubic-bezier(.2,.8,.2,1), width .55s cubic-bezier(.2,.8,.2,1), height .55s cubic-bezier(.2,.8,.2,1), border-radius .55s',
      top: '0px', left: '0px', width: '100vw', height: '100vh', borderRadius: '0px', border: 'none',
    });
    setTimeout(() => ui().classList.add('show'), 400);
  }

  function exitZoom() {
    const f = document.getElementById('live-frame');
    const holder = document.getElementById('live-holder');
    ui().classList.remove('show');
    document.documentElement.style.overflow = '';
    if (!f || !zoomed) { zoomed = false; return; }
    zoomed = false;
    const r = holder ? holder.getBoundingClientRect() : { top: 0, left: 0, width: innerWidth, height: 720 };
    Object.assign(f.style, {
      top: r.top + 'px', left: r.left + 'px', width: r.width + 'px', height: r.height + 'px',
      borderRadius: '8px', boxShadow: 'none',
    });
    setTimeout(() => {
      if (zoomed) return;
      Object.assign(f.style, {
        position: '', zIndex: '', top: '', left: '', margin: '', transition: '', boxShadow: '',
        width: '100%', height: '720px', border: '1px solid #ddd', borderRadius: '8px',
      });
    }, 560);
  }

  function hookEsc(win) {
    // 焦點在預覽裡面時也要能用 Esc 離開
    try {
      if (win.__escHooked) return;
      win.__escHooked = true;
      win.addEventListener('keydown', e => { if (e.key === 'Escape') exitZoom(); });
    } catch (e) {}
  }
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && zoomed) exitZoom(); });

  function frameDoc() {
    const f = document.getElementById('live-frame');
    return f ? (f.contentDocument || f.contentWindow.document) : null;
  }

  setInterval(() => {
    const box = document.querySelector('#live-src textarea');
    if (!box) return;
    const raw = box.value || '';
    const nl = raw.indexOf('\\n');
    if (nl < 0) return;
    const [id, state] = raw.slice(0, nl).split('|');
    const code = raw.slice(nl + 1);
    const doc = frameDoc();
    if (!doc) return;
    const frame = document.getElementById('live-frame');

    // 新的一次生成，或預覽視窗被換掉了：重新開始
    if (id !== runId || frame.dataset.run !== id) {
      runId = id; written = ''; closed = false;
      frame.dataset.run = id;
      frame.dataset.zoomedOnce = '';
      doc.open();
      hookEsc(frame.contentWindow);
    }
    if (closed) return;

    if (code.startsWith(written)) {
      const delta = code.slice(written.length);
      if (delta) { doc.write(delta); written = code; }
    } else {
      // 內容對不上（例如最後整理過空白）：整份重寫
      doc.open(); doc.write(code); written = code;
    }

    // 網頁開始浮現（body 裡出現第一個元素）時，自動放大到全螢幕，每次生成只觸發一次
    if (!frame.dataset.zoomedOnce && doc.body && doc.body.children.length > 0 && autoZoomOn()) {
      frame.dataset.zoomedOnce = '1';
      enterZoom();
    }

    const statusEl = document.getElementById('zoom-status');
    if (statusEl) {
      const st = document.querySelector('#gen-status');
      statusEl.textContent = st ? st.innerText.trim() : '';
    }

    if (state === 'run') {
      // 自動捲到最新長出來的地方
      try { frame.contentWindow.scrollTo(0, doc.documentElement.scrollHeight); } catch (e) {}
    } else if (state === 'done') {
      doc.close();
      closed = true;
      try { frame.contentWindow.scrollTo({ top: 0, behavior: 'smooth' }); } catch (e) {}
    }
  }, 120);
}
"""

FULLSCREEN_JS = """
() => {
  const f = document.querySelector('#preview iframe');
  if (!f) { alert('請先生成網頁'); return; }
  (f.requestFullscreen || f.webkitRequestFullscreen).call(f);
}
"""

CSS = """
.title-row { display: flex; align-items: center; gap: 10px; }
.title-row h1 { margin: 0; font-size: 1.8rem; }

#sk-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 8px; }
.sk-btn {
    display: inline-flex; align-items: center; gap: 6px; box-sizing: border-box;
    height: 34px; margin: 0; padding: 0 10px; font-size: 14px; line-height: 1; cursor: pointer;
    border: 1px solid var(--border-color-primary); border-radius: 8px;
    background: var(--background-fill-primary); color: var(--body-text-color);
}
.sk-btn:hover { background: var(--background-fill-secondary); }
.sk-btn.active { border-color: var(--color-accent); color: var(--color-accent); background: var(--color-accent-soft); }
.sk-btn svg { flex-shrink: 0; }
.sk-sep { width: 1px; height: 24px; margin: 0 2px; background: var(--border-color-primary); }
#sk-size {
    box-sizing: border-box; height: 34px; margin: 0; padding: 0 8px; font-size: 14px; border-radius: 8px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-primary); color: var(--body-text-color);
}
#sk-stage {
    display: flex; justify-content: center; overflow: hidden;
    border: 1px solid var(--border-color-primary); border-radius: 8px; background: #ffffff;
}
.sk-hint { margin-top: 6px; font-size: 12px; color: var(--body-text-color-subdued); }

#live-code textarea, #think-box textarea {
    font-family: ui-monospace, "Cascadia Code", Menlo, Consolas, monospace !important;
    font-size: 13px !important; line-height: 1.5 !important;
}
#think-box textarea { color: var(--body-text-color-subdued) !important; }
#live-src { display: none !important; }

/* 全螢幕時右上角的狀態與離開按鈕 */
#zoom-bar {
    position: fixed; top: 14px; right: 14px; z-index: 2147483600;
    display: flex; align-items: center; gap: 10px;
    padding: 6px 6px 6px 14px; border-radius: 999px;
    background: rgba(17, 24, 39, .78); color: #f9fafb; font-size: 13px;
    backdrop-filter: blur(6px);
    opacity: 0; transform: translateY(-8px); pointer-events: none;
    transition: opacity .25s, transform .25s;
}
#zoom-bar.show { opacity: .35; transform: none; pointer-events: auto; }
#zoom-bar.show:hover { opacity: 1; }
#zoom-bar button {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
    padding: 6px 12px; border: none; border-radius: 999px;
    background: rgba(255, 255, 255, .14); color: inherit; font-size: 13px;
}
#zoom-bar button:hover { background: rgba(255, 255, 255, .26); }
#gen-status p { margin: 4px 0; font-size: 14px; color: var(--body-text-color-subdued); }

#preview iframe:fullscreen {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    border-radius: 0 !important;
}
"""


def toggle_canvas(mode):
    """放大：隱藏右側預覽，畫布會自動撐滿整個寬度"""
    return gr.update(visible=(mode != "放大"))


# ---------------- 介面 ----------------

# Gradio 6 起 css / head 改放在 launch()；這裡兩種版本都支援
GR6 = int(gr.__version__.split(".")[0]) >= 6
STYLE_KW = {"css": CSS, "head": FABRIC_HEAD}

with gr.Blocks(title="草圖轉網頁", **({} if GR6 else STYLE_KW)) as demo:
    gr.HTML(f'<div class="title-row">{svg("pencil", size=30)}<h1>手繪草圖 → 網頁</h1></div>')
    gr.Markdown("在畫布上畫草圖、加文字方塊，或上傳手繪照片當底圖，再按「生成網頁」。")
    with gr.Row():
        with gr.Column(scale=1, min_width=420):  # 左右 1:1，畫布比較好畫
            canvas_mode = gr.Radio(
                ["標準", "放大"], value="標準", label="畫布大小",
                info="放大時會暫時隱藏右側預覽；按「生成網頁」後自動切回標準",
            )
            gr.HTML(CANVAS_HTML)
            sketch_data = gr.Textbox(visible=False)
            extra = gr.Textbox(label="額外要求（選填）", placeholder="例如：咖啡廳官網、深色主題、主色用綠色")
            gen_btn = gr.Button("生成網頁", variant="primary", icon=icon_file("sparkles", "#ffffff"))
            refine_txt = gr.Textbox(label="修改指令", placeholder="例如：按鈕改成圓角、加一個聯絡表單")
            refine_btn = gr.Button("依指令修改", icon=icon_file("refresh", "#6b7280"))
            file_out = gr.File(label="下載 HTML")
        with gr.Column(scale=1) as preview_col:
            fs_btn = gr.Button("全螢幕展示（按 Esc 離開）", icon=icon_file("maximize", "#6b7280"))
            status = gr.Markdown("尚未生成", elem_id="gen-status")
            live_src = gr.Textbox(elem_id="live-src", show_label=False, container=False)
            auto_fs = gr.Checkbox(value=True, label="開始繪製時自動放大成全螢幕（Esc 離開）",
                                  elem_id="auto-fs", interactive=True)
            with gr.Tabs(selected="preview") as tabs:
                with gr.Tab("預覽", id="preview"):
                    view = gr.HTML(elem_id="preview")
                with gr.Tab("產出過程", id="live"):
                    with gr.Accordion("模型思考過程", open=False):
                        think_box = gr.Textbox(show_label=False, lines=8, max_lines=8,
                                               autoscroll=True, interactive=False, elem_id="think-box")
                    live_code = gr.Textbox(label="程式碼即時產出", lines=26, max_lines=26,
                                           autoscroll=True, interactive=False, elem_id="live-code")
                with gr.Tab("程式碼", id="code"):
                    code = gr.Code(language="html")

    demo.load(None, None, None, js=CANVAS_JS)
    demo.load(None, None, None, js=LIVE_JS)
    canvas_mode.change(toggle_canvas, canvas_mode, preview_col)
    # 先在瀏覽器把畫布匯出成圖片、切回標準模式，再送給模型
    STREAM_OUT = [status, think_box, live_code, code, view, file_out, tabs, live_src]
    gen_btn.click(
        lambda d: (d, "標準"), [sketch_data], [sketch_data, canvas_mode], js=EXPORT_JS
    ).then(generate, [sketch_data, extra], STREAM_OUT)
    refine_btn.click(refine, [code, refine_txt], STREAM_OUT)
    fs_btn.click(None, None, None, js=FULLSCREEN_JS)

demo.queue().launch(server_name="0.0.0.0", server_port=7860, **(STYLE_KW if GR6 else {}))