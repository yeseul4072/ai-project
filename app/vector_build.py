import os
import re
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings  # 최신 권장 방식
from langchain_community.vectorstores import FAISS  # 최신 경로

load_dotenv()  # .env 파일에서 환경변수 로드

def split_by_subtitle(text: str, screen_id: str):
    # 서브 타이틀별로 분류
    pattern = r"\[(.*?)\]\s*([\s\S]*?)(?=\n\[|$)"
    chunks = []
    matches = re.findall(pattern, text)
    for section_title, section_text in matches:
        content = section_text.strip()
        if content:
            chunks.append(
                Document(
                    page_content=content,
                    metadata={"screen_id": screen_id, "section": section_title}
                )
            )
    return chunks

def build_vector_db():
    screen_text = """
    [개요]
    매입/매출 전표를 기준으로 매입/매출 세금계산서를 발행/취소하고 관리합니다.

    [주요처리흐름]
    • 세금계산서 내역 조회: 조회조건에 해당하는 세금계산서 내역을 조회합니다.
    • 세금계산서발행: 
        - 증빙구분이 매출인 경우 정발행, 매입인 경우 역발행만 발행 가능합니다.
        - 증빙일자와 전기일자가 같을 때만 발행 가능합니다.
        - 오늘 일자가 10일보다 이전인 경우 증빙일자가 전월, 당월인 내역 발행 가능하며 10일보다 이후인 경우 당월인 내역만 발행 가능
        - 임의발행인 경우 발행대기상태에서만 발행완료로 발행상태 변경 가능하며, 발행완료로 변경하여야 부가세신고내역에서 조회됩니다.
    • 세금계산서발행취소: 
        - 증빙구분이 매출인 경우 정발행, 매입인 경우 역발행만 발행 취소 가능합니다.
        - 국세청 미전송된 내역 중 전송상태가 미승인, 오류, 반송인 내역만 취소 가능합니다.
        - 임의발행인 경우 발행완료상태에서만 발행대기로 발행상태 변경 가능합니다. 
        - 이미 취소 진행중인 경우 취소할 수 없습니다. 
    • 특이사항: 
        - 증빙일자, 전기일자가 다른 내역 발행이 필요한 경우 공통코드 VAT_PBLT_EXCP에 계정 등록 후 발행할 수 있습니다 
        - 발행 기준이 되는 일 변경이 필요한 경우 운영팀에 문의바랍니다

    [버튼설명]
    • 초기화: 화면을 처음 열었던 상태로 초기화 처리합니다.
    • 조회: 검색조건에 해당하는 세금계산서 내역을 조회합니다.
    • 저장: 발행구분을 저장합니다.
    • 세금계산서발행: 선택한 행에 대해 발행 요청을 합니다.
    • 세금계산서발행취소: 선택한 행에 대해 발행 취소 요청을 합니다.
    """

    screen_id = "invoice_issue"
    docs = split_by_subtitle(screen_text, screen_id)

    embedding_model = OpenAIEmbeddings()  # .env에 키가 있어야 함
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("vector_db/screens")
    print("벡터 DB 저장 완료!")

if __name__ == "__main__":
    build_vector_db()
