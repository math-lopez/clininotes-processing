import os
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import tempfile
from pathlib import Path
from typing import Optional

# from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# load_dotenv()

app = FastAPI(title="CliniNotes Processing Service")

INTERNAL_TOKEN = os.getenv("PROCESSING_SERVICE_INTERNAL_TOKEN")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

if not INTERNAL_TOKEN:
    raise RuntimeError("PROCESSING_SERVICE_INTERNAL_TOKEN não configurado")

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

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(
            f"[model] Carregando WhisperModel "
            f"(size={WHISPER_MODEL_SIZE}, device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE})"
        )
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[model] Modelo carregado com sucesso")
    return _model


def extract_highlights(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences[:5]


def extract_next_steps(text: str) -> str:
    if not text.strip():
        return ""

    return "Revisar a sessão e validar manualmente os pontos principais antes de salvar no prontuário."


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_size": WHISPER_MODEL_SIZE,
        "device": WHISPER_DEVICE,
    }


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
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file.content_type}",
        )

    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    tmp_path: Optional[str] = None

    try:
        contents = await file.read()

        file_size_bytes = len(contents)
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"Arquivo muito grande. Máximo permitido: {MAX_FILE_SIZE_MB} MB",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        print(
            f"[process-audio] Iniciando transcrição "
            f"(sessionId={sessionId}, filename={file.filename}, size_mb={file_size_mb:.2f})"
        )

        model = get_model()

        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
        )

        transcript_parts = []
        for segment in segments:
            if segment.text:
                transcript_parts.append(segment.text.strip())

        transcript = " ".join(part for part in transcript_parts if part).strip()
        highlights = extract_highlights(transcript)
        next_steps = extract_next_steps(transcript)

        print(
            f"[process-audio] Transcrição concluída "
            f"(sessionId={sessionId}, language={getattr(info, 'language', None)})"
        )

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

    except HTTPException as http_error:
        return JSONResponse(
            status_code=http_error.status_code,
            content={
                "success": False,
                "error": http_error.detail,
            },
        )
    except Exception as e:
        print(f"[process-audio] Erro inesperado: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_error:
            print(f"[cleanup] Falha ao remover arquivo temporário: {cleanup_error}")