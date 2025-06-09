FROM python:3.11-slim

# This tells girder_worker to enable gpu if possible
LABEL com.nvidia.volumes.needed=nvidia_driver

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install timm

RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels

# Install the latest version of large_image.  This can be disabled if the
# latest version we need has had an official release
RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/girder/large_image && \
    cd large_image && \
    # git checkout zarr-channel-axis && \
    pip install .[all] -r requirements-dev.txt --find-links https://girder.github.io/large_image_wheels

COPY . /opt/main
WORKDIR /opt/main
RUN python -m pip install -e . --find-links https://girder.github.io/large_image_wheels

WORKDIR /opt/main/histomicstk_extras

RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint RegisterImages --help
RUN python -m slicer_cli_web.cli_list_entrypoint AnnotationFeatures --help

# This makes the results show up in a more timely manner
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
