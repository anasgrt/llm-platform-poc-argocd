"""Embedded chat UI rendering."""

import os

from .config import QWEN3_URL, client


_MODEL_NAME_CACHE: str | None = None


def get_model_name() -> str:
    """Fetch the served model id from qwen3-server and cache it.

    llama.cpp returns the model file path (e.g. /models/Qwen3-4B-Q4_K_M.gguf);
    we strip the directory and the .gguf suffix so the badge stays compact.
    """
    global _MODEL_NAME_CACHE
    if _MODEL_NAME_CACHE:
        return _MODEL_NAME_CACHE
    try:
        r = client.get(f"{QWEN3_URL}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            mid = str(data[0].get("id", ""))
            name = os.path.basename(mid)
            if name.endswith(".gguf"):
                name = name[: -len(".gguf")]
            if name:
                _MODEL_NAME_CACHE = name
                return name
    except Exception:
        pass
    return "model unavailable"


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Log Analysis — __MODEL_NAME__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:16px 24px;border-bottom:1px solid #1e2130;display:flex;align-items:center;gap:12px}
header h1{font-size:18px;font-weight:500;color:#fff}
header span{font-size:12px;background:#2a4a3a;color:#4ade80;padding:2px 10px;border-radius:99px}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:720px;padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.7;white-space:pre-wrap}
.msg.user{background:#1e3a5f;align-self:flex-end;color:#bfdbfe}
.msg.bot{background:#1e2130;align-self:flex-start;border:1px solid #2a2d3a}
.msg.bot .sources{margin-top:12px;padding-top:10px;border-top:1px solid #2a2d3a;font-size:12px;color:#888}
.input-bar{padding:16px 72px 16px 24px;border-top:1px solid #1e2130;display:flex;gap:10px}
.input-bar input{flex:1;background:#1a1d2e;border:1px solid #2a2d3a;color:#fff;padding:12px 16px;border-radius:10px;font-size:14px;outline:none}
.input-bar input:focus{border-color:#3b82f6}
.input-bar button{background:#3b82f6;color:#fff;border:none;padding:12px 24px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:500;min-width:118px}
.input-bar button:disabled{opacity:.4;cursor:not-allowed}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #444;border-top-color:#3b82f6;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <h1>Log Analysis Platform</h1>
  <span>__MODEL_NAME__</span>
  <button id="clearBtn" style="margin-left:auto;background:#2a2d3a;color:#888;border:1px solid #3a3d4a;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px">Clear</button>
</header>
<div class="chat" id="chat">
  <div class="msg bot">Ready. Ask me about your logs — errors, patterns, root causes, correlations.<br><br>
Examples:<br>• What errors keep recurring?<br>• Why are pods crashlooping?<br>• Summarize the NGINX 5xx errors<br>• What happened around 03:14 UTC?</div>
</div>
<form class="input-bar" id="form" autocomplete="off">
  <input id="q" placeholder="Ask about your logs..." autofocus>
  <button id="btn" type="submit">Analyze</button>
</form>
<script>
const chat=document.getElementById('chat'),form=document.getElementById('form'),q=document.getElementById('q'),btn=document.getElementById('btn');
const STORAGE_KEY='log_chat_history';
const INPUT_HISTORY_KEY='log_input_history';
const MAX_HISTORY=50;
const MAX_INPUT_HISTORY=100;
let busy=false;
let inputHistory=JSON.parse(localStorage.getItem(INPUT_HISTORY_KEY)||'[]');
let historyIndex=-1;
let savedDraft='';

function loadHistory(){
  try{
    const raw=localStorage.getItem(STORAGE_KEY);
    if(!raw) return;
    const history=JSON.parse(raw);
    if(!Array.isArray(history)||!history.length) return;
    chat.innerHTML='';
    for(const item of history){
      chat.innerHTML+=`<div class="msg user">${esc(item.q)}</div>`;
      const bubble=document.createElement('div');
      bubble.className='msg bot';
      const body=document.createElement('span');
      body.textContent=item.a;
      bubble.appendChild(body);
      if(item.src){
        const src=document.createElement('div');
        src.className='sources';
        src.textContent=item.src;
        bubble.appendChild(src);
      }
      chat.appendChild(bubble);
    }
    chat.scrollTop=chat.scrollHeight;
  }catch(_){}
}

function saveHistory(){
  const msgs=Array.from(chat.children);
  const history=[];
  let lastUser=null;
  for(const el of msgs){
    if(el.classList.contains('user')){
      lastUser=el.textContent;
    }else if(el.classList.contains('bot')&&lastUser){
      const body=el.querySelector('span')||el;
      const srcEl=el.querySelector('.sources');
      history.push({q:lastUser,a:body.textContent,src:srcEl?srcEl.textContent:''});
      lastUser=null;
    }
  }
  if(history.length>MAX_HISTORY) history.splice(0,history.length-MAX_HISTORY);
  localStorage.setItem(STORAGE_KEY,JSON.stringify(history));
}

// Walk the visible chat to build the {role, content} list the server expects.
// Only completed user→bot pairs count; the just-appended in-progress user msg
// has no bot pair yet, so it's correctly excluded.
function buildHistoryPayload(){
  const out=[];
  let pendingUser=null;
  for(const el of chat.children){
    if(el.classList.contains('user')){
      pendingUser=el.textContent;
    }else if(el.classList.contains('bot')&&pendingUser){
      const body=el.querySelector('span')||el;
      const answer=body.textContent.trim();
      if(answer){
        out.push({role:'user',content:pendingUser});
        out.push({role:'assistant',content:answer});
      }
      pendingUser=null;
    }
  }
  return out;
}

form.addEventListener('submit',send);
btn.addEventListener('click',send);
q.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    if(form.requestSubmit) form.requestSubmit(); else send(e);
  }
  if(e.key==='ArrowUp'&&inputHistory.length){
    e.preventDefault();
    if(historyIndex===-1) savedDraft=q.value;
    if(historyIndex<inputHistory.length-1) historyIndex++;
    q.value=inputHistory[inputHistory.length-1-historyIndex];
  }
  if(e.key==='ArrowDown'){
    e.preventDefault();
    if(historyIndex>0){ historyIndex--; q.value=inputHistory[inputHistory.length-1-historyIndex]; }
    else if(historyIndex===0){ historyIndex=-1; q.value=savedDraft; }
  }
});
async function send(e){
  if(e) e.preventDefault();
  if(busy)return;
  const text=q.value.trim(); if(!text)return;
  inputHistory.push(text);
  if(inputHistory.length>MAX_INPUT_HISTORY) inputHistory.splice(0,inputHistory.length-MAX_INPUT_HISTORY);
  localStorage.setItem(INPUT_HISTORY_KEY,JSON.stringify(inputHistory));
  historyIndex=-1; savedDraft='';
  busy=true; q.value=''; btn.disabled=true;
  chat.innerHTML+=`<div class="msg user">${esc(text)}</div>`;
  const bubble=document.createElement('div');
  bubble.className='msg bot';
  const body=document.createElement('span');
  let status=document.createElement('span');
  status.innerHTML='<div class="spinner"></div> Retrieving logs...';
  bubble.appendChild(body); bubble.appendChild(status);
  chat.appendChild(bubble); chat.scrollTop=chat.scrollHeight;
  let answer='';
  let sourcesText='';
  try{
    const history=buildHistoryPayload();
    const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text,history})});
    if(!r.ok){
      let detail=''; try{ detail=(await r.json()).detail||''; }catch(_){}
      throw new Error(detail||('HTTP '+r.status));
    }
    const reader=r.body.getReader(), decoder=new TextDecoder();
    let buf='';
    while(true){
      const {value,done}=await reader.read();
      if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\\n'); buf=lines.pop();
      for(const line of lines){
        if(!line.trim()) continue;
        let evt; try{ evt=JSON.parse(line); }catch(_){ continue; }
        if(evt.type==='token'){
          if(status){ status.remove(); status=null; }
          answer+=evt.data;
          body.textContent=answer;
        } else if(evt.type==='status'){
          status.textContent=evt.data;
        } else if(evt.type==='done'){
          if(evt.sources&&evt.sources.length){
            const src=document.createElement('div');
            src.className='sources';
            sourcesText=`Based on ${evt.num_chunks_used} log chunks from: ${[...new Set(evt.sources.map(s=>s.source))].join(', ')}`;
            src.textContent=sourcesText;
            bubble.appendChild(src);
          }
        } else if(evt.type==='error'){
          bubble.innerHTML='Error: '+esc(evt.data);
        }
      }
      chat.scrollTop=chat.scrollHeight;
    }
  }catch(e){bubble.innerHTML='Error: '+esc(e.message)}
  busy=false; btn.disabled=false; q.focus(); chat.scrollTop=chat.scrollHeight;
  saveHistory();
}
window.send=send;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
loadHistory();

document.getElementById('clearBtn').addEventListener('click',()=>{
  if(!confirm('Clear chat history?')) return;
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(INPUT_HISTORY_KEY);
  inputHistory=[]; historyIndex=-1; savedDraft='';
  chat.innerHTML=`<div class="msg bot">Ready. Ask me about your logs — errors, patterns, root causes, correlations.<br><br>Examples:<br>• What errors keep recurring?<br>• Why are pods crashlooping?<br>• Summarize the NGINX 5xx errors<br>• What happened around 03:14 UTC?</div>`;
});
</script>
</body></html>"""




def render_ui_html() -> str:
    return CHAT_HTML.replace("__MODEL_NAME__", get_model_name())
