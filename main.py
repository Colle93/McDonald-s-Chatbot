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
    1) Appende messaggio utente
    2) Avvia run Assistant
    3) Polla fino a fine
    4) Se completed → prendi la risposta dagli step (message_creation)
       Se failed/cancelled → mostra last_error (e il chiamante farà un fallback)
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

    # ---- gestione esiti non OK (con dettaglio) ----
    if run.status != "completed":
        last_err = getattr(run, "last_error", None)
        if last_err:
            # tipicamente ha campi .code e .message
            code = getattr(last_err, "code", None)
            msg  = getattr(last_err, "message", None)
            print(f"[RUN][ERROR] code={code} message={msg}")
            return f"[assistant failed] code={code} message={msg}"
        print("[RUN][ERROR] no last_error provided")
        return f"[assistant failed] status={run.status}"

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

    # Fallback: ultimo assistant nel thread
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
        answer = "Non ho trovato una risposa dall'assistente."
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
    Crea un nuovo thread Assistants. Restituisce SEMPRE 200 per non far fallire l'API tool.
    Se c'è un errore, torna {"thread_id": null, "error": "..."} così VF può gestirlo.
    """
    try:
        # stile stabile beta.* (non dipende da namespace "chat")
        thread = client.beta.threads.create()
        print(f"[THREAD] created: {thread.id}")
        return JSONResponse(content={"thread_id": thread.id, "error": None}, status_code=200)
    except Exception as e:
        # NON 500: 200 + payload di errore, così Voiceflow non mostra [API tool] failed
        msg = str(e)
        print("[ERROR] /start:", msg)
        return JSONResponse(content={"thread_id": None, "error": msg}, status_code=200)

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
