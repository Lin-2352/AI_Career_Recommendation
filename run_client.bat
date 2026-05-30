@echo off
setlocal

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is required but was not found in PATH.
    exit /b 1
)

if not exist .env (
    if exist .env.example copy .env.example .env >nul
)

docker compose version >nul 2>nul
if errorlevel 1 (
    docker-compose --version >nul 2>nul
    if errorlevel 1 (
        echo Docker Compose is required but was not found.
        exit /b 1
    )
    docker-compose up --build
) else (
    docker compose up --build
)
