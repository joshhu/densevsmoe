"""FastAPI 進入點：serve 前端靜態檔與 API。"""
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import extract
from .models import MODEL_CATALOG, ModelManager

app = FastAPI(title="Dense vs MoE 視覺化")
manager = ModelManager()

STATIC_DIR = Path(__file__).parent.parent / "static"


class LoadReq(BaseModel):
    model_id: str


class InferReq(BaseModel):
    dense_model: str
    moe_model: str
    sentence: str


@app.get("/api/models")
def list_models():
    return [{**asdict(m),
             "state": manager.status.get(m.id, {}).get("state", "idle")}
            for m in MODEL_CATALOG]


@app.post("/api/load", status_code=202)
def load_model(req: LoadReq):
    try:
        result = manager.request_load(req.model_id)
    except KeyError:
        raise HTTPException(400, "未知的模型 ID（僅支援策展清單內的模型）") from None
    if result.get("state") == "error":
        raise HTTPException(409, result.get("detail", "載入失敗"))
    return result


@app.get("/api/status")
def status():
    return manager.status


@app.post("/api/infer")
def infer(req: InferReq):
    if not req.sentence.strip():
        raise HTTPException(400, "請輸入句子")
    dense = manager.get("dense", req.dense_model)
    moe = manager.get("moe", req.moe_model)
    if dense is None or moe is None:
        raise HTTPException(400, "模型尚未載入，請先載入兩側模型")
    d_info = manager.catalog_entry(req.dense_model)
    m_info = manager.catalog_entry(req.moe_model)
    try:
        d = extract.extract_dense(dense[0], dense[1], req.sentence)
        m = extract.extract_moe(moe[0], moe[1], req.sentence)
    except extract.SentenceTooLong as e:
        raise HTTPException(400, str(e)) from None
    d.update(model_id=req.dense_model, params_per_token=d_info.params_active)
    m.update(model_id=req.moe_model,
             active_params_per_token=m_info.params_active,
             total_params=m_info.params_total)
    return {"dense": d, "moe": m}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
