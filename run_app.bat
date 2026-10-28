@echo off
title Auto MPG - Streamlit App
echo ===================================================
echo Demarrage de l'application Auto MPG avec Streamlit
echo ===================================================
echo Verification et installation des dependances...
pip install -r requirements.txt
echo Lancement du serveur...
streamlit run app.py
pause
