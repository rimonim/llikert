# M0 container probe (NOT the release image): CUDA runtime + pinned llama-cpp-python wheel
# built against llama.cpp v0.4.1, a stdlib /health stub, model read from /repository.
# build context: repo root
#   docker build -f spikes/container/Dockerfile -t llikert-probe:m0 .
#   docker run --rm --gpus all -p 8080:8080 -v $PWD/models:/repository:ro \
#     -e LLIKERT_MODEL=/repository/qwen2.5-0.5b-instruct-q8_0.gguf llikert-probe:m0
#   curl -i localhost:8080/health
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m app
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY build/wheels/cuda/llama_cpp_python-0.3.35-py3-none-linux_x86_64.whl /tmp/
# --no-deps: the probe loads libllama directly and never imports the binding
RUN pip install --no-cache-dir --no-deps /tmp/llama_cpp_python-0.3.35-py3-none-linux_x86_64.whl \
    && rm /tmp/*.whl
COPY spikes/native.py spikes/container/health_stub.py /app/
USER app
WORKDIR /app
EXPOSE 8080
CMD ["python3", "health_stub.py"]
