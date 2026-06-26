FROM python:3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY midi2edgelist/package.json /app/midi2edgelist/package.json
COPY midi2edgelist/package-lock.json /app/midi2edgelist/package-lock.json
RUN cd /app/midi2edgelist && npm ci

CMD ["bash"]