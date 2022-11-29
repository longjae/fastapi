from fastapi import APIRouter
from fastapi import UploadFile
import os, sys
from service import homeservice

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
)

router = APIRouter()

@router.get("/")
def home():
    return "hello world"

@router.post("/file")
async def upload_file(file: UploadFile):
    if (os.path.isdir("uploads/") == False):
        os.mkdir("uploads/")
    file_dir = "uploads/" + file.filename
    with open(file_dir, "wb+") as f:
        f.write(file.file.read())
            
    pred_class = homeservice.predict(file_dir)   
    print(pred_class)
    return {"Message": pred_class}