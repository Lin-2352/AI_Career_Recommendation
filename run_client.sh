#!/usr/bin/env sh
set -eu

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required but was not found in PATH."
    exit 1
fi

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi

if docker compose version >/dev/null 2>&1; then
    docker compose up --build
elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose up --build
else
    echo "Docker Compose is required but was not found."
    exit 1
fi
