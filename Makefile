.PHONY: setup test demo demo-no-docker readme-check terraform-test

setup:
	uv sync --frozen --all-extras

test:
	uv run --frozen pytest

demo:
	uv run --frozen dsgate demo --out build/demo --train sagemaker-local

demo-no-docker:
	uv run --frozen dsgate demo --out build/demo --train process

readme-check:
	uv run --frozen dsgate readme --check

terraform-test:
	cd infra && terraform fmt -check -recursive && terraform init -backend=false -input=false && terraform validate && terraform test
