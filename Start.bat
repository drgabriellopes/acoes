@echo off
REM =====================================================
REM  Synapse Colosseum - Inicializador (Windows)
REM =====================================================

setlocal

cd /d "%~dp0"

echo.
echo ========================================
echo   Synapse Colosseum - Iniciando...
echo ========================================
echo.

REM --- Verifica Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo        Instale Python 3.10+ em https://www.python.org/downloads/
    pause
    exit /b 1
)

REM --- Cria venv se necessario ---
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Criando ambiente virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar venv.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Ambiente virtual ja existe.
)

REM --- Ativa venv ---
echo [2/4] Ativando venv...
call venv\Scripts\activate.bat

REM --- Instala/atualiza dependencias ---
echo [3/4] Instalando dependencias...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias.
    pause
    exit /b 1
)

REM --- Verifica .env ---
if not exist ".env" (
    if exist ".env.example" (
        echo [AVISO] .env nao encontrado. Copiando .env.example...
        copy /Y ".env.example" ".env" >nul
        echo        Edite .env e adicione suas API keys antes de usar os modelos.
    ) else (
        echo [AVISO] .env nao encontrado e sem .env.example.
    )
)

REM --- Garante diretorios ---
if not exist "logs" mkdir logs
if not exist "sessions" mkdir sessions
if not exist "cache" mkdir cache
if not exist "podcasts" mkdir podcasts

REM --- Inicia servidor ---
echo [4/4] Iniciando servidor Flask em http://localhost:5000
echo.
echo ========================================
echo   Abra http://localhost:5000 no browser
echo   Pressione CTRL+C para parar
echo ========================================
echo.

REM Abre navegador em 3 segundos (nao bloqueia)
start "" /min cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

python app.py

endlocal
pause
