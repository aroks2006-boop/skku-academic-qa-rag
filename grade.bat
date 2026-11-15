@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 블라인드 채점 화면을 엽니다. 끝내려면 이 창을 닫으세요.
start "" /min cmd /c "timeout /t 5 /nobreak > nul & start http://localhost:8503"
python -m streamlit run eval\human_grade.py --server.headless true --server.port 8503 --browser.gatherUsageStats false
pause
