ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS build

# Maxwell (5.0) through Blackwell (12.0), plus PTX for forward compatibility with newer GPUs
ARG CUDA_ARCHS="50 60 70 75 80 86 89 90 100 120"

WORKDIR /src
COPY cudaPi.cu .
RUN gencode=""; \
    for arch in ${CUDA_ARCHS}; do gencode="${gencode} -gencode arch=compute_${arch},code=sm_${arch}"; done; \
    last=$(echo ${CUDA_ARCHS} | awk '{print $NF}'); \
    nvcc -O2 -Wno-deprecated-gpu-targets ${gencode} \
         -gencode arch=compute_${last},code=compute_${last} \
         cudaPi.cu -o cudaPi

FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU_VERSION}

LABEL org.opencontainers.image.source="https://github.com/Chandan-M-N/carl" \
      org.opencontainers.image.description="Carl - CUDA load tester for Kubernetes"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/carl
COPY --from=build --chmod=755 /src/cudaPi /opt/carl/cudaPi
COPY --chmod=755 gpuTest.sh /opt/carl/gpuTest.sh

# the "ubuntu" user (uid 1000) ships with the ubuntu24.04 base image
USER 1000:1000

ENTRYPOINT ["/opt/carl/gpuTest.sh"]
CMD ["512"]
