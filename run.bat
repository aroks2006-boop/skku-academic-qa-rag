@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".env" (
  echo [오류] .env 파일이 없습니다. setup.bat 을 먼저 실행해 주세요.
  pause
  exit /b 1
)
if not exist "chroma" (
  echo [오류] 검색 인덱스가 없습니다. data\raw 에 PDF를 넣고 setup.bat 을 먼저 실행해 주세요.
  pause
  exit /b 1
)
echo SKKU 학사 Q^&A 앱을 시작합니다. 잠시 후 브라우저가 열립니다. 끝내려면 이 창을 닫으세요.
start "" /min cmd /c "timeout /t 5 /nobreak > nul & start http://localhost:8501"
python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
pause
