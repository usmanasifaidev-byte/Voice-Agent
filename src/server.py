from fastapi import FastAPI
import logging
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from app.api.state import router as state_router
from app.api.voice import router as voice_router
from app.api.webrtc import router as webrtc_router

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

app.include_router(state_router, prefix="/api")
app.include_router(voice_router, prefix="/api")
app.include_router(webrtc_router, prefix="/api")
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join("public", "chat.html"))

@app.get("/chat.html")
def chat_page():
    return FileResponse(os.path.join("public", "chat.html"))
