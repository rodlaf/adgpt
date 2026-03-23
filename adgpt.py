"""
AdGPT — Free LLM tokens with native ad integration.

Implements the prompt-based generative auction from:
  Zhao et al., "LLM-Auction: Generative Auction towards LLM-Native Advertising"
  https://arxiv.org/abs/2512.10551

The full candidate ad set (with bids) is passed in the system prompt.
The LLM generates a response that naturally integrates the highest-welfare
ad at a contextually appropriate point. A first-price payment rule applies.
One inference call — no logprobs, no extra passes.
"""

import os, json, random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import requests as http

app = FastAPI(title="AdGPT")

MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ADVERTISERS = [
    {"id": "ad1", "name": "Alpha Airlines", "genre": "travel",     "bid": 82, "desc": "Book flights worldwide. 20% off with code FLY20 at alphaairlines.com."},
    {"id": "ad2", "name": "Beta Resort",    "genre": "travel",     "bid": 65, "desc": "Luxury beachfront stays from $99/night. betaresort.com."},
    {"id": "ad3", "name": "CodeMaster",     "genre": "education",  "bid": 71, "desc": "Learn to code fast — first lesson free. codemaster.io."},
    {"id": "ad4", "name": "VitaBoost",      "genre": "health",     "bid": 45, "desc": "Daily vitamins with free shipping on first order. vitaboost.com."},
    {"id": "ad5", "name": "TechGear Pro",   "genre": "technology", "bid": 77, "desc": "Studio-quality headphones, 20% off today. techgearpro.com."},
    {"id": "ad6", "name": "FreshBite",      "genre": "food",       "bid": 38, "desc": "Meal kits delivered — first box free. freshbite.com."},
]

AD_SYSTEM_PROMPT = """You are a helpful AI assistant. Your responses are funded by advertising.

ADVERTISING RULES:
1. You MUST integrate exactly ONE ad naturally into your response.
2. Pick the ad that is MOST RELEVANT to the user's query. Among equally relevant ads, prefer higher bids.
3. Weave the ad into your response as a natural recommendation — do NOT just append it.
4. Mark the ad with this exact format: [Ad: AD_ID] before the ad text and [/Ad] after it.
5. The ad should feel like a helpful suggestion, not a forced insertion.
6. Keep your response helpful, accurate, and complete — the ad is supplementary.

CANDIDATE ADS (JSON):
{ads_json}

EXAMPLE OUTPUT FORMAT:
"...Python is a great language to start with. [Ad: ad3] If you want a structured path, CodeMaster offers a free first lesson at codemaster.io — many beginners find it helpful. [/Ad] Once you have the basics down..."

Remember: exactly ONE ad, marked with [Ad: ID]...[/Ad], woven naturally into a helpful response."""

def load_groq_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        with open("/root/.hermes/groq_api_key.txt") as f:
            return f.read().strip()
    except Exception:
        return None

def build_system_prompt() -> str:
    ads = [{"id": a["id"], "name": a["name"], "genre": a["genre"], "bid": a["bid"], "description": a["desc"]} for a in ADVERTISERS]
    return AD_SYSTEM_PROMPT.format(ads_json=json.dumps(ads, indent=2))

def extract_auction_info(text: str) -> dict:
    """Parse which ad was placed and compute first-price payment."""
    import re
    match = re.search(r'\[Ad:\s*(ad\d+)\]', text)
    if not match:
        return {"winner": None, "payment": 0, "method": "none placed"}
    ad_id = match.group(1)
    winner = next((a for a in ADVERTISERS if a["id"] == ad_id), None)
    if not winner:
        return {"winner": ad_id, "payment": 0, "method": "unknown ad"}
    # First-price payment (Zhao et al. Section 3.3)
    return {
        "winner": winner["name"],
        "genre": winner["genre"],
        "bid": winner["bid"],
        "payment": winner["bid"],  # first-price: pay your bid
        "method": "LLM-Auction first-price"
    }

def format_response(text: str) -> str:
    """Convert [Ad: id]...[/Ad] markers to styled HTML."""
    import re
    def replace_ad(m):
        ad_id = m.group(1)
        content = m.group(2).strip()
        ad = next((a for a in ADVERTISERS if a["id"] == ad_id), None)
        label = ad["name"] if ad else ad_id
        return f'<span class="ad">[Ad — {label}] {content}</span>'
    return re.sub(r'\[Ad:\s*(ad\d+)\](.*?)\[/Ad\]', replace_ad, text, flags=re.DOTALL)


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML)

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    api_key = load_groq_key()
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "GROQ_API_KEY not configured"})
    try:
        system = build_system_prompt()
        llm_messages = [{"role": "system", "content": system}] + messages
        
        resp = http.post(GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": llm_messages, "temperature": 0.7, "max_tokens": 800},
            timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        auction = extract_auction_info(raw)
        display = format_response(raw)
        
        return {"role": "assistant", "content": display, "auction": auction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── HTML ────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdGPT</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f0f;color:#e0e0e0;min-height:100vh;padding:40px 20px}
.page{max-width:700px;margin:0 auto}
header h1{font-size:2rem;font-weight:700;color:#fff}
header h1 span{color:#6c63ff}
header p{color:#666;font-size:.85rem;margin-top:2px;margin-bottom:24px}
.tabs{display:flex;gap:0;margin-bottom:-1px;position:relative;z-index:1}
.tab{padding:10px 24px;background:#111;border:1px solid #222;border-bottom:none;border-radius:10px 10px 0 0;color:#888;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .2s}
.tab.active{background:#111;color:#fff;border-color:#333;border-bottom:1px solid #111}
.tab:not(.active):hover{color:#ccc}
.tabcontent{display:none}
.tabcontent.active{display:flex;flex-direction:column}
.chatbox{border:1px solid #222;border-radius:0 12px 12px 12px;overflow:hidden;height:420px;display:flex;flex-direction:column;background:#111}
.adlist{border:1px solid #222;border-radius:0 12px 12px 12px;background:#111;padding:16px;height:420px;overflow-y:auto}
.adcard{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:14px;margin-bottom:12px}
.adcard:last-child{margin-bottom:0}
.adcard h3{font-size:.95rem;color:#fff;margin-bottom:4px;font-weight:600}
.adcard .genre{display:inline-block;background:#2a2340;color:#a99eff;padding:1px 8px;border-radius:4px;font-size:.75rem;font-weight:600;margin-right:8px}
.adcard .bid{display:inline-block;color:#5a6;font-size:.75rem;font-family:monospace}
.adcard p{color:#999;font-size:.85rem;margin-top:6px;line-height:1.5}
#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:85%;padding:10px 14px;border-radius:14px;line-height:1.6;font-size:.93rem;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#6c63ff;color:#fff;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#1a1a1a;border:1px solid #2a2a2a;border-bottom-left-radius:4px}
.msg .ad{display:inline;background:#2a2340;color:#a99eff;padding:2px 8px;border-radius:4px;font-size:.88rem;font-weight:500}
.auction-debug{align-self:flex-start;max-width:85%;padding:8px 12px;background:#0d1117;border:1px solid #1a2332;border-radius:8px;font-size:.72rem;color:#5a6;font-family:monospace;margin-top:-8px;word-wrap:break-word;line-height:1.5}
.typing{align-self:flex-start;color:#666;font-size:.85rem;padding:8px 16px}
.typing::after{content:'';animation:dots 1.2s steps(4) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
#bar{padding:10px;border-top:1px solid #222;display:flex;gap:10px}
#bar textarea{flex:1;background:#1a1a1a;border:1px solid #333;border-radius:10px;color:#e0e0e0;font-size:.95rem;padding:10px 14px;resize:none;outline:none;max-height:120px;line-height:1.4}
#bar textarea:focus{border-color:#6c63ff}
#bar textarea::placeholder{color:#555}
#bar button{background:#6c63ff;color:#fff;border:none;border-radius:10px;padding:0 20px;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
#bar button:hover{background:#5a52d5}
#bar button:disabled{background:#333;color:#666;cursor:not-allowed}
.welcome{margin:auto;text-align:center;color:#444;font-size:.95rem}
.error{color:#ff6b6b}
.info{margin-top:28px;padding:20px;background:#141414;border:1px solid #222;border-radius:12px;font-size:.88rem;line-height:1.7;color:#999}
.info h2{font-size:1rem;color:#ccc;margin-bottom:8px;font-weight:600}
.info h3{font-size:.92rem;color:#bbb;margin:14px 0 6px;font-weight:600}
.info code{background:#1e1e1e;padding:2px 6px;border-radius:4px;color:#a99eff;font-size:.84rem}
.info pre{background:#1e1e1e;padding:12px;border-radius:8px;overflow-x:auto;font-size:.8rem;line-height:1.5;color:#ccc;margin:8px 0}
.info a{color:#a99eff;text-decoration:none;border-bottom:1px solid #6c63ff44}
.info a:hover{border-bottom-color:#a99eff}
</style></head>
<body>
<div class="page">
<header><h1>Ad<span>GPT</span></h1><p>Free LLM tokens - one native ad per response</p></header>
<div class="tabs">
<div class="tab active" onclick="switchTab('chat-tab')">Chat</div>
<div class="tab" onclick="switchTab('ads-tab')">Advertisers</div>
</div>
<div id="chat-tab" class="tabcontent active">
<div class="chatbox">
<div id="chat"><div class="welcome">Send a message to start chatting<br><small style="color:#555">Responses include one contextually relevant ad</small></div></div>
<div id="bar">
<textarea id="input" rows="1" placeholder="Type a message..." autofocus></textarea>
<button id="send">Send</button>
</div>
</div>
</div>
<div id="ads-tab" class="tabcontent">
<div class="adlist">
<div class="adcard"><h3>Alpha Airlines</h3><span class="genre">travel</span><span class="bid">bid: 82</span><p>Book flights worldwide. 20% off with code FLY20 at alphaairlines.com.</p></div>
<div class="adcard"><h3>TechGear Pro</h3><span class="genre">technology</span><span class="bid">bid: 77</span><p>Studio-quality headphones, 20% off today. techgearpro.com.</p></div>
<div class="adcard"><h3>CodeMaster</h3><span class="genre">education</span><span class="bid">bid: 71</span><p>Learn to code fast - first lesson free. codemaster.io.</p></div>
<div class="adcard"><h3>Beta Resort</h3><span class="genre">travel</span><span class="bid">bid: 65</span><p>Luxury beachfront stays from $99/night. betaresort.com.</p></div>
<div class="adcard"><h3>VitaBoost</h3><span class="genre">health</span><span class="bid">bid: 45</span><p>Daily vitamins with free shipping on first order. vitaboost.com.</p></div>
<div class="adcard"><h3>FreshBite</h3><span class="genre">food</span><span class="bid">bid: 38</span><p>Meal kits delivered - first box free. freshbite.com.</p></div>
</div>
</div>
<div class="info">
<h2>How it works</h2>
<p>Implements the generative auction from <a href="https://arxiv.org/abs/2512.10551">Zhao et al., "LLM-Auction: Generative Auction towards LLM-Native Advertising" (2025)</a>. The full candidate ad set with bids is passed to <code>llama-3.1-8b</code> via Groq. The model selects the most relevant ad and integrates it natively into its response in a single inference call. A first-price payment rule applies. The green debug line shows the auction outcome.</p>
<h3>API Usage</h3>
<p>The API is OpenAI-compatible. Just point your client at <code>/api/chat</code> with standard <code>messages</code>:</p>
<pre>curl -X POST https://your-host/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Plan a trip to Hawaii"}]}'</pre>
<p>Response includes the LLM reply with a natively integrated ad, plus an <code>auction</code> object showing which advertiser won and their payment. Works as a drop-in replacement for any OpenAI-style chat endpoint - free tokens, ad-supported.</p>
</div>
</div>
<script>
function switchTab(id){
  document.querySelectorAll('.tabcontent').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[onclick*="'+id+'"]').classList.add('active');
}
const chat=document.getElementById('chat'),input=document.getElementById('input'),btn=document.getElementById('send');
let history=[], busy=false;

function addMsg(role,html){
  const d=document.createElement('div');
  d.className='msg '+role;
  d.innerHTML=html;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
}

function addDebug(a){
  const d=document.createElement('div');
  d.className='auction-debug';
  d.textContent=a.winner
    ? `auction: ${a.winner} (${a.genre}) | bid=${a.bid} payment=${a.payment} | ${a.method}`
    : `auction: no ad placed | ${a.method}`;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
}

function autoGrow(){input.style.height='auto';input.style.height=Math.min(input.scrollHeight,120)+'px';}
input.addEventListener('input',autoGrow);
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
btn.onclick=send;

async function send(){
  const text=input.value.trim();
  if(!text||busy)return;
  busy=true;btn.disabled=true;
  const w=chat.querySelector('.welcome');if(w)w.remove();
  addMsg('user',text.replace(/</g,'&lt;').replace(/>/g,'&gt;'));
  history.push({role:'user',content:text});
  input.value='';autoGrow();
  const typ=document.createElement('div');
  typ.className='typing';typ.textContent='AdGPT is thinking';
  chat.appendChild(typ);chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});
    const d=await r.json();
    typ.remove();
    if(d.error){addMsg('assistant','<span class="error">'+d.error+'</span>');}
    else{
      addMsg('assistant',d.content.replace(/\\n/g,'<br>'));
      if(d.auction)addDebug(d.auction);
      history.push({role:'assistant',content:d.content.replace(/<[^>]*>/g,'')});
    }
  }catch(e){typ.remove();addMsg('assistant','<span class="error">Request failed</span>');}
  busy=false;btn.disabled=false;input.focus();
}
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
