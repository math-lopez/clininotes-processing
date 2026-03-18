import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CliniNotes Processing Service")

INTERNAL_TOKEN = os.getenv("PROCESSING_SERVICE_INTERNAL_TOKEN")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

if not INTERNAL_TOKEN:
    raise RuntimeError("PROCESSING_SERVICE_INTERNAL_TOKEN não configurado")

# Carrega o modelo uma vez no startup do processo
model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

ALLOWED_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
}


def extract_highlights(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences[:5]


def extract_next_steps(text: str) -> str:
    if not text.strip():
        return ""
    # Regra simples inicial. Depois você troca por um algoritmo melhor.
    return "Revisar a sessão e validar manualmente os pontos principais antes de salvar no prontuário."


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    sessionId: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization inválido")

    token = authorization.replace("Bearer ", "", 1)
    if token != INTERNAL_TOKEN:
        raise HTTPException(status_code=403, detail="Token interno inválido")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Tipo de arquivo não suportado: {file.content_type}")

    suffix = Path(file.filename or "audio.bin").suffix or ".bin"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,
        )

        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        transcript = " ".join(part for part in transcript_parts if part).strip()
        highlights = extract_highlights(transcript)
        next_steps = extract_next_steps(transcript)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "sessionId": sessionId,
                "text": transcript,
                "highlights": highlights,
                "next_steps": next_steps,
                "language": getattr(info, "language", None),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass