import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 환경 변수 로드 (.env에 OPENAI_API_KEY 포함)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. 엑셀 파일 로딩 (경로는 사용 환경에 따라 변경)
EXCEL_PATH = "screens.xlsx"
df = pd.read_excel(EXCEL_PATH)

# 3. DataFrame에서 Document 객체 리스트 생성
documents = []
for _, row in df.iterrows():
    screen_id = str(row["화면ID"])
    section = str(row["섹션"])
    content = str(row["설명내용"]).strip()

    if content:
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "screen_id": screen_id,
                    "section": section
                }
            )
        )

# 4. 벡터화 및 저장
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("vector_db/screens")

print("✅ FAISS 벡터 DB 저장 완료: vector_db/screens")
