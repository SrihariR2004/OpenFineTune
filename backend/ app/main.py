from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid
import ray
from .training import run_openrlhf_training
from .mlflow_utils import init_mlflow

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

ray.init(address="auto", ignore_reinit_error=True)
init_mlflow()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train/")
async def train(
    background_tasks: BackgroundTasks,
    dataset_file: UploadFile = File(...),
    training_method: str = Form(...),
    base_model: str = Form(...),
):
    job_id = str(uuid.uuid4())
    dataset_path = UPLOAD_DIR / f"{job_id}_{dataset_file.filename}"
    with dataset_path.open("wb") as buffer:
        shutil.copyfileobj(dataset_file.file, buffer)

    background_tasks.add_task(run_openrlhf_training, str(dataset_path), training_method, base_model, job_id)
    return {"message": "Training started", "job_id": job_id}
