
.PHONY: help
.DEFAULT_GOAL := help

COMPOSE_FILE_OPT = -f ./docker/docker-compose.yml
DOCKER_COMPOSE_CMD = docker compose $(COMPOSE_FILE_OPT)


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build backend image
	$(call log, Building backend image (using host SSH keys)...)
	DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE_CMD) build

up: ## Start services in foreground
	$(call log, Starting services in detached mode...)
	$(DOCKER_COMPOSE_CMD) up -d backend

ps: ## Show containers status
	$(call log, Showing containers status...)
	$(DOCKER_COMPOSE_CMD) ps

down: ## Stop and remove containers
	$(call log, Stopping and removing containers...)
	$(DOCKER_COMPOSE_CMD) down

logs: ## Tail logs for backend container
	$(call log, Tailing of logs for backend...)
	$(DOCKER_COMPOSE_CMD) logs -f backend

run: ## Run a command inside a backend container. Usage: make run cmd='poetry run python3 tp1/app/main.py'
	$(DOCKER_COMPOSE_CMD) run -w /project \
		backend \
		$(cmd)

run-tp: ## Run a the entrypoint of a tp. Usage: make run-tp tp='path-to-tp'
	$(MAKE) run cmd="poetry run python3 tp1/app/main.py"

	