@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo [1/3] 필요한 패키지를 설치합니다...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [오류] 패키지 설치에 실패했습니다. Python 3.11 이상이 설치되어 있는지 확인해 주세요.
  pause
  exit /b 1
)
if not exist ".env" (
  copy ".env.example" ".env" > nul
  echo [2/3] .env 파일을 만들었습니다. 메모장이 열리면 OPENAI_API_KEY= 뒤에 키를 붙여넣고 저장한 뒤 닫아 주세요.
  notepad ".env"
) else (
  echo [2/3] .env 파일이 이미 있습니다.
)
echo [3/3] 문서를 인덱싱합니다. 스캔본 OCR이 처음이면 20~40분, 약 3달러가 듭니다...
python ingest.py
if errorlevel 1 (
  echo [오류] 인덱싱에 실패했습니다. data\raw 에 PDF가 있는지, API 키가 맞는지 확인해 주세요.
  pause
  exit /b 1
)
echo 준비 완료. 이제 run.bat 을 더블클릭하면 앱이 열립니다.
pause
