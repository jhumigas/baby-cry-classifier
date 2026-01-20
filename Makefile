OUTPUT_DIR = .
SRC_DIR = src/baby_cry_classifier

.PHONY: help
help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } ' $(MAKEFILE_LIST)


.PHONY: install
install: ## install dependencies
	uv sync --locked --all-extras --dev

.PHONY: check-bandit
check-bandit: ## code vulnerability security scan
	uvx bandit -c pyproject.toml -r .

# Numpy safety check error are ignored du to sktime dependencies. Turn numpy to 1.22.0 to resolve safety check when
# this issue will be resolved https://github.com/alan-turing-institute/sktime/discussions/2037
.PHONY: check-safety
check-safety: ## dependencies scan with safety
	uv pip freeze | uvx safety check -i 44715 -i 44716 -i 44717 --stdin

.PHONY: pre-commit
pre-commit: ## run pre-commit hooks
	uvx pre-commit run --all-files

.PHONY: install-pre-commit
install-pre-commit: ## install pre-commit hooks
	uvx pre-commit install

.PHONY: detect-secrets
detect-secrets: ## detect secrets in code
	uvx detect-secrets scan $(SRC_DIR) --all-files

.PHONY: format
format: ## format code
	uvx ruff format .

.PHONY: lint
lint: ## lint code
	uvx ruff check .

.PHONY: unit-tests
unit-tests:  ## run unit tests
	uv run pytest tests/unit

.PHONY: start-dev-fast-app
start-app: ## start fastapi app in dev mode
	uv run fastapi dev src/baby_cry_classifier/serve.py

.PHONY: train-model
train-model: ## train model
	uv run python scripts/train.py

.PHONY: test-predict
test-predict: ## test model prediction
	uv run python scripts/predict.py --input data/sample/sample.json

.PHONY:
docker-build: ## build docker image
	docker build -t baby-cry-classifier-image .

.PHONY: docker-run
docker-run: ## run docker container
	docker run -d -p 9696:8000 --name baby-cry-classifier-container baby-cry-classifier-image

.PHONY: docker-stop
docker-stop: ## stop docker container
	docker stop baby-cry-classifier-container && docker rm baby-cry-classifier-container