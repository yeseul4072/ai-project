
### 선행 작업
₩₩₩
python3 --version # python 3.8이상
pip3 --version
₩₩₩
- Python 3.8 이상, pip 설치 확인
- 설치 후 레포지토리 clone

### 가상환경 활성화
₩₩₩
python -m venv venv # 생성
source venv/bin/activate # 실행
₩₩₩
- 프로젝트 루트 디렉토리에서 가상환경을 만든다
- 성공 시 터미널 앞에 (venv) 가 보임
- 가상환경은 **Python과 pip의 “독립된 실행 공간”**을 만들어줌 -> 가상환경을 실행하지 않고 설치하면 **글로벌 환경(시스템 전체)**에 라이브러리가 설치됨

### 라이브러리 설치
₩₩₩
pip3 install -r requirements.txt
₩₩₩
- 가상환경 실행 후 라이브러리 설치

### env 파일 생성 
₩₩₩
OPENAI_API_KEY=sk-xxx-your-api-key
₩₩₩
- openapi 키 환경파일에 입력(.gitignore에 포함되어 있어 레포지토리에 올라가지않음)

### 서버 실행
₩₩₩
uvicorn main:app --reload
₩₩₩
- http://127.0.0.1:8000/chat 로 접속!