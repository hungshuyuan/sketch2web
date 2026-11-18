ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG LLAMA_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp . && git checkout ${LLAMA_REF}
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/ /app/
ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
