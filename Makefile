# Makefile para PROTEUS

.PHONY: help build run demo evolution interactive clean shell web-build web-run web-stop web-logs

help:
	@echo "PROTEUS - Comandos disponibles:"
	@echo ""
	@echo "Simulación original:"
	@echo "  make build       - Construir imagen Docker"
	@echo "  make demo        - Ejecutar demo visual"
	@echo "  make evolution   - Ejecutar experimento evolutivo"
	@echo "  make interactive - Abrir shell interactivo"
	@echo "  make clean       - Limpiar archivos generados"
	@echo "  make shell       - Abrir bash en contenedor"
	@echo ""
	@echo "Aplicación web:"
	@echo "  make web-build   - Construir aplicación web"
	@echo "  make web-run     - Ejecutar aplicación web (http://localhost:3000)"
	@echo "  make web-stop    - Detener aplicación web"
	@echo "  make web-logs    - Ver logs de la aplicación web"

build:
	docker compose build

demo: build
	@echo "Ejecutando demo visual..."
	@mkdir -p output
	docker compose run --rm proteus python visual_demo_docker.py
	@echo "Imágenes guardadas en output/"

evolution: build
	@echo "Ejecutando experimento evolutivo..."
	@mkdir -p experiment_results
	docker compose run --rm proteus-evolution
	@echo "Resultados guardados en experiment_results/"

interactive: build
	@echo "Abriendo shell interactivo..."
	docker compose run --rm proteus-interactive

shell: build
	docker compose run --rm proteus-interactive /bin/bash

clean:
	@echo "Limpiando archivos generados..."
	rm -rf output/*.png
	rm -rf experiment_results/*
	rm -rf __pycache__
	rm -rf proteus/__pycache__
	rm -rf proteus/*/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

stop:
	docker compose down

logs:
	docker compose logs -f

# Web application commands
web-build:
	docker compose -f docker-compose.web.yml build

web-run:
	@echo "Iniciando PROTEUS Web..."
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	docker compose -f docker-compose.web.yml up

web-stop:
	docker compose -f docker-compose.web.yml down

web-logs:
	docker compose -f docker-compose.web.yml logs -f