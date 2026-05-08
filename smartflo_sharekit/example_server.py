"""FastAPI SmartFlo bridge server for apsagent / rnf_agnet profiles."""

import os
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from example_bot import run_bot


SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env", override=False)

app = FastAPI(title="SmartFlo Pipecat Bridge")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "profile": os.getenv("SMARTFLO_AGENT_PROFILE", "rnf_agnet"),
        "ws_endpoint": "/ws/smartflo",
    }


@app.websocket("/ws/smartflo")
async def smartflo_endpoint(websocket: WebSocket):
    """
    One WebSocket connection = one phone call.
    SmartFlo opens this connection when a call comes in.
    """
    await websocket.accept()
    logger.info(f"SmartFlo connection from {websocket.client}")

    try:
        await run_bot(websocket)
    except WebSocketDisconnect:
        logger.info("SmartFlo call ended (WebSocket disconnected)")
    except Exception as e:
        logger.error(f"SmartFlo session error: {e}")
        raise
