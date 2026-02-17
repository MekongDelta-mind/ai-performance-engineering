FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip "setuptools<81" wheel

# CUDA-enabled torch used by FP4 benchmark scripts.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1+cu130

# Plot dependencies used by local analysis scripts.
RUN python -m pip install numpy matplotlib

# DeepGEMM (open-source) for FP8xFP4 benchmarks.
RUN git clone --depth 1 --recursive --shallow-submodules https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM \
    && python -m pip install --no-build-isolation /opt/DeepGEMM

# nvbandwidth (open-source) for dedicated bandwidth bundle.
RUN git clone --depth 1 https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth \
    && cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/nvbandwidth/build -j"$(nproc)" \
    && cp /opt/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth

WORKDIR /workspace

CMD ["bash"]
