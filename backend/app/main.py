import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from app.routers import detect, file_detect, health
from app.services.detector_service import DetectorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_template_dir = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_template_dir)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AIGC Detector...")
    detector = DetectorService()
    detector.load_models()
    app.state.detector = detector
    logger.info("AIGC Detector ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AIGC Text Detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(detect.router)
app.include_router(file_detect.router)
app.include_router(health.router)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    template = _env.get_template("index.html")
    return template.render(request=request)
