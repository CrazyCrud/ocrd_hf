ARG DOCKER_BASE_IMAGE=ocrd/core-cuda-torch:2024
FROM $DOCKER_BASE_IMAGE

ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/CrazyCrud/ocrd_hf" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="YourOrg" \
    org.opencontainers.image.title="ocrd_hf" \
    org.opencontainers.image.description="OCR-D wrapper for HuggingFace OCR recognition" \
    org.opencontainers.image.source="https://github.com/YourOrg/ocrd_hf" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=$DOCKER_BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Avoid unpredictable HOME paths
ENV XDG_DATA_HOME /usr/local/share
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

WORKDIR /build/ocrd_hf

# Copy everything
COPY . .

# Also ensure the ocrd-tool.json is at root (if needed)
COPY ocrd_hf/ocrd-tool.json ./ocrd-tool.json

# Install system dependencies if needed (e.g. for image I/O, fonts, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Prepackage tool manifests so OCR-D CLI can find them
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json && \
    ocrd ocrd-tool ocrd-tool.json dump-module-dirs > $(dirname $(ocrd bashlib filename))/ocrd-all-module-dir.json

# Install Python package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Clean build-time deps to reduce image size
RUN apt-get -y remove --auto-remove wget && \
    apt-get clean && \
    rm -rf /build/ocrd_hf

WORKDIR /data
VOLUME /data

# Default command
CMD ["ocrd-hf-recognize", "--help"]
