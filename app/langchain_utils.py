# app/langchain_utils.py
from langchain.chat_models import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

api_key = os.getenv("OPENAI_API_KEY")

# 서버 실행 중에 메모리 유지
memory = ConversationBufferMemory(return_messages=True)

def get_openai_response():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=api_key)
    
    response = llm.invoke("넌 누구니?")
    return response.content

def get_chat_response(user_input: str):
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