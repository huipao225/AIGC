FROM python:3.12-slim

WORKDIR /code

# Use China mirrors
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX}
ENV HF_ENDPOINT=https://hf-mirror.com

# Debian China mirrors
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch via SJTU mirror
RUN pip install --no-cache-dir torch \
    -i https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host mirror.sjtu.edu.cn \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# Application code
COPY ./app /code/app

RUN mkdir -p /code/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
