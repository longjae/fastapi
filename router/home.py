from fastapi import APIRouter
from fastapi import UploadFile
from typing import List

router = APIRouter()

@router.get("/")
def home():
    return "hello world"

@router.post("/file")
async def upload_file(file: List[UploadFile] = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        return {"filename": file.filename}