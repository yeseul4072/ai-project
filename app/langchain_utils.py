import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. 환경 변수 로드 및 API 키 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. 모델 및 벡터스토어 초기화
# GPT-3.5 Turbo 모델
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=api_key) 
# OpenAIEmbeddings는 OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터(숫자 배열)로 변환하는 클래스
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
# FAISS 벡터 데이터베이스를 로컬에서 불러옴
vectorstore = FAISS.load_local(
    "vector_db/screens",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
) 

# vectorstore: FAISS에서 로드된 벡터 데이터베이스 객체
# as_retriever(): LangChain 내부에서 사용할 수 있도록 벡터스토어 → 리트리버 API로 래핑해주는 함수
# 상위 3개 문서를 검색
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# 3. 프롬프트 분리: 쿼리 리라이팅용 + 문서 기반 응답용
# (1) 쿼리 리라이팅 프롬프트 (retriever용) - GPT가 사용자의 질문을 재구성할 수 있도록 도와줌
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 대화를 참고하여 사용자의 질문을 명확하게 다시 표현해주세요."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# (2) 문서 기반 응답 프롬프트 (LLM용) - FAISS에서 검색된 문서 목록을 context로 받아 GPT가 더 정확한 답변을 할 수 있게 함
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "ERP 화면 설명서를 참고하여 사용자 질문에 답변해주세요. 참고 문서:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# 4. 체인 구성
# 질문과 대화 이력을 GPT로 보내서 질문 리라이팅 -> retriever가 리라이팅된 질문으로 벡터 검색 수행 
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=query_prompt
)

# 검색된 문서(context)와 사용자의 질문을 기반으로 답변을 생성
qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# 위 두 체인을 조합한 최종 QA 체인
retrieval_chain: Runnable = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain
)

# 5. 대화 메모리
# 이전 질문/답변을 저장하여 대화의 맥락을 유지 -> “그거” 같은 모호한 표현도 이해 가능
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 사용자의 질문 → 벡터 DB 검색 → 유사 문서 3개 추출 → GPT에게 문서와 함께 질문 전달 → 답변 생성
def get_chat_response(user_input: str):
    result = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": memory.chat_memory.messages
    })
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(result["answer"])
    return result["answer"]

# 벡터DB 없이 LLM + Langchain 이용한 응답 메소드
def get_simple_chat_response(user_input: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=api_key)

    # chat prompt 
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "이 시스템은 재무 관련 질문에 답변할 수 있습니다"), # 역할, 내용
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])

     # 체인 연결
    conversation = ConversationChain(
        memory=memory,
        prompt=chat_prompt,
        llm=llm,
        verbose=False
    )

    # 입력값 처리
    return conversation.predict(input=user_input) # 이전 히스토리 + 새 질문 LLM에 전달하여 답변 생성

def get_openai_response():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=api_key)
    
    response = llm.invoke("넌 누구니?")
    return response.content