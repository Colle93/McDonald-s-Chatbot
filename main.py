import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import openai as openai_pkg

# =========================
# ENV
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASSISTANT_ID = os.environ.get("ASSISTANT_ID")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")
if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID env var")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FastAPI
# =========================
app = FastAPI(title="Voiceflow-Assistant Bridge (stable beta.threads)")

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
    Flusso stabile con Assistants:
      1) append user message
      2) create run
      3) poll fino a completed
      4) usa run steps per recuperare i message_id creati
      5) estrae il testo dal messaggio assistant
    """

    # 1) append message utente
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # 2) avvia run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    # 3) poll finché termina
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        status = getattr(run, "status", "unknown")
        print(f"[RUN] status={status}")
        if status in ("completed", "failed", "cancelled", "expired", "requires_action"):
            break
        await asyncio.sleep(0.6)

    if run.status != "completed":
        return f"Errore: run status {run.status}"

    # 4) steps -> message_id creati dall'assistente in questa run
    steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run.id,
        order="asc",
        limit=50
    )

    created_message_ids = []
    for s in steps.data:
        if getattr(s, "type", None) == "message_creation":
            try:
                mid = s.step_details.message_creation.message_id
                if mid:
                    created_message_ids.append(mid)
            except Exception:
                pass

    # 5) recupera i messaggi assistant creati e prendi il testo
    answer = None
    if created_message_ids:
        for mid in created_message_ids:
            m = client.beta.threads.messages.retrieve(
                thread_id=thread_id,
                message_id=mid
            )
            if getattr(m, "role", "") == "assistant" and getattr(m, "content", None):
                for part in m.content:
                    txt = getattr(part, "text", None)
                    if txt and getattr(txt, "value", None):
                        answer = txt.value
                        break
            if answer:
                break

    # Fallback: prendi l'ultimo assistant nel thread (se necessario)
    if not answer:
        msgs = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=25
        )
        for m in msgs.data:
            if getattr(m, "role", "") == "assistant" and getattr(m, "content", None):
                for part in m.content:
                    txt = getattr(part, "text", None)
                    if txt and getattr(txt, "value", None):
                        answer = txt.value
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
    return JSONResponse(
        content={"ok": True, "openai_version": getattr(openai_pkg, "__version__", "unknown")},
        status_code=200
    )

@app.get("/version")
def version():
    return JSONResponse(content={"version": "stable-beta-threads-v1"}, status_code=200)

@app.get("/vf-test")
def vf_test():
    # per testare rapidamente Voiceflow: Capture -> response.response = "pong"
    return JSONResponse(content={"response": "pong"}, status_code=200)

@app.get("/start")
def start_conversation():
    """
    Crea un nuovo thread.
    In Voiceflow: Capture -> response.thread_id  (o response.body.thread_id a seconda del workspace)
    """
    try:
        thread = client.beta.threads.create()
        print(f"[THREAD] created: {thread.id}")
        return JSONResponse(content={"thread_id": thread.id}, status_code=200)
    except Exception as e:
        print("[ERROR] /start:", repr(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Body atteso:
      { "thread_id": "<thd_...>", "message": "<domanda_utente>" }
    Ritorna:
      { "response": "<testo>" }
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
        print("[ERROR] /chat:", repr(e))
        return JSONResponse(content={"response": f"Errore interno: {e}"}, status_code=200)
