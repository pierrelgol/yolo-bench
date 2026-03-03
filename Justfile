# Show available recipes.
default: help

# Print recipe help.
help:
    @just --list

# Create the local virtual environment once.
venv:
    @if [ -d .venv ]; then \
        echo ".venv already exists"; \
    else \
        uv venv .venv; \
    fi

# Fetch the coco128 dataset into the top-level dataset directory.
fetch:
    uv run --with ultralytics python3 src/dataset-fetcher/fetch_coco128.py

# Launch the Qt target labeler.
label:
    uv run python3 src/targets-labels/label_targets.py

# Remove Python caches.
clean:
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    rm -rf .mypy_cache .pytest_cache .ruff_cache

# Remove generated artifacts, including the venv and downloaded dataset.
fclean: clean
    rm -rf .venv dataset/coco128 dataset/targets dataset/class.txt
