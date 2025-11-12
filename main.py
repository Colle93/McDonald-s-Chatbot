import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# =========================
# Config & Client
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASSISTANT_ID = os.environ.get("ASSISTANT_ID")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")
if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID env var")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Voiceflow-Assistant Bridge")

# CORS aperto (utile per tool diversi; Voiceflow tipicamente chiama da server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# =========================
# Models
# =========================
class ChatRequest(BaseModel):
    thread_id: str
    message: str

# =========================
# Utils
# =========================
async def run_assistant_and_get_reply(thread_id: str, user_message: str) -> str:
    """
    1) Appende il messaggio dell'utente al thread
    2) Avvia la run dell'assistente
    3) Polla finché la run è 'completed' (o termina per errore)
    4) Recupera il/i message_id creati dalla run tramite i run-steps
    5) Estrae il testo dal/i messaggio/i dell'assistente (fallback: ultimo assistant nel thread)
    """
    # 1) append message
    client.chat.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # 2) create run
    run = client.chat.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)

    # 3) poll
    while True:
        run = client.chat.runs.retrieve(thread_id=thread_id, run_id=run.id)
        status = getattr(run, "status", "unknown")
        print(f"[RUN] status={status}")
        if status in ("completed", "failed", "cancelled", "expired", "requires_action"):
            break
        await asyncio.sleep(0.6)

    if run.status != "completed":
        # Non mandiamo errore 5xx a Voiceflow: restituiamo comunque JSON leggibile
        return f"Errore: run status {run.status}"

    # 4) steps → message_id creati da QUESTA run
    steps = client.chat.runs.steps.list(thread_id=thread_id, run_id=run.id, order="asc", limit=50)
    created_message_ids = []
    for s in steps.data:
        # gli step di creazione messaggio possono essere identificati da type
        stype = getattr(s, "type", None)
        if stype == "message_creation":
            try:
                mid = s.step_details.message_creation.message_id
                if mid:
                    created_message_ids.append(mid)
            except Exception:
                pass

    # 5) estrai il testo dai messaggi assistant creati da questa run
    answer = None
    if created_message_ids:
        for mid in created_message_ids:
            m = client.chat.messages.retrieve(thread_id=thread_id, message_id=mid)
            if getattr(m, "role", "") == "assistant" and getattr(m, "content", None):
                for part in m.content:
                    # i content parts hanno .type (es. "output_text") e un .text.value
                    text_obj = getattr(part, "text", None)
                    if text_obj and getattr(text_obj, "value", None):
                        answer = text_obj.value
                        break
            if answer:
                break

    # Fallback: cerca l'ultimo messaggio assistant nel thread se per qualche motivo non troviamo dai steps
    if not answer:
        msgs = client.chat.messages.list(thread_id=thread_id, order="desc", limit=25)
        for m in msgs.data:
            if getattr(m, "role", "") == "assistant" and getattr(m, "content", None):
                for part in m.content:
                    text_obj = getattr(part, "text", None)
                    if text_obj and getattr(text_obj, "value", None):
                        answer = text_obj.value
                        break
            if answer:
                break

    if not answer:
        answer = "Non ho trovato una risposta dall'assistente."
    print(f"[ASSISTANT] {answer[:200]}{'...' if len(answer) > 200 else ''}")
    return answer

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return JSONResponse(content={"ok": True}, status_code=200)

@app.get("/version")
def version():
    return JSONResponse(content={"version": "chat-steps-v3"}, status_code=200)

@app.get("/vf-test")
def vf_test():
    # endpoint di test per Voiceflow: deve catturare response.response -> "pong"
    return JSONResponse(content={"response": "pong"}, status_code=200)

@app.get("/start")
def start_conversation():
    """
    Crea un nuovo thread per Voiceflow. In VF cattura:
    response.body.thread_id  (se disponibile)
    oppure, a seconda della tua versione: response.thread_id  o  response.body.response.thread_id
    (Nel tuo workspace hai usato finora response.thread_id → adattati a quello che VF mostra)
    """
    thread = client.chat.threads.create()
    print(f"[THREAD] created: {thread.id}")
    return JSONResponse(content={"thread_id": thread.id}, status_code=200)

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Body atteso:
    {
      "thread_id": "<thd_...>",
      "message":   "<domanda_utente>"
    }

    Ritorna SEMPRE JSON:
    { "response": "<testo assistant o errore leggibile>" }
    """
    thread_id = (req.thread_id or "").strip()
    user_msg  = (req.message or "").strip()

    if not thread_id:
        raise HTTPException(status_code=400, detail="Missing thread_id")
    if not user_msg:
        return JSONResponse(content={"response": "Dimmi qualcosa da inviare al bot."}, status_code=200)

    try:
        answer = await run_assistant_and_get_reply(thread_id, user_msg)
        return JSONResponse(content={"response": answer}, status_code=200)
    except Exception as e:
        # log e risposta "safe" per Voiceflow
        print(f"[ERROR] /chat: {e}")
        return JSONResponse(content={"response": f"Errore interno: {e}"}, status_code=200)
