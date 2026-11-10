FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip awscli && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy scipy trimesh smplx cupy-cuda12x
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY weekend_run/ /root/weekend_run/
COPY entrypoint.py /opt/entrypoint.py
ENTRYPOINT ["python3", "/opt/entrypoint.py"]
