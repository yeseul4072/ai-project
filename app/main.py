from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import Body
from pydantic import BaseModel
from dotenv import load_dotenv
import os 
from app.langchain_utils import get_openai_response, get_chat_response
from fastapi.staticfiles import StaticFiles

load_dotenv()  # .env 파일 로드

app = FastAPI()


# 요청 바디 모델 정의
class ChatRequest(BaseModel):
    user_input: str
    

@app.get("/chat_test")
def chat_test():
    answer = get_openai_response()
    return JSONResponse(content={"message": answer}, media_type="application/json; charset=utf-8")

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    answer = get_chat_response(request.user_input)
    return JSONResponse(content={"message": answer}, media_type="application/json; charset=utf-8")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
