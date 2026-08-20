FROM --platform=linux/amd64 python:3.8.18-slim-bookworm@sha256:e796941013b10bb53a0924d8705485a1afe654bbbc6fe71d32509101e44b6414

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/opt/loss-of-plasticity-a6b79580d85f3025bdb601566d3627c5f489f13b

WORKDIR /opt/qualification
COPY requirements.lock requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    --requirement requirements.lock

COPY fetch_source.py fetch_source.py
RUN python fetch_source.py

COPY Dockerfile Dockerfile.source
COPY requirements.in requirements.in
COPY qualification-plan.json qualification-plan.json
COPY verify_runtime.py verify_runtime.py
RUN python verify_runtime.py

USER 65532:65532
ENTRYPOINT ["python", "/opt/qualification/verify_runtime.py"]
