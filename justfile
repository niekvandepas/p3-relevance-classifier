default: build push

build:
    docker buildx build --platform linux/amd64 -t niekpas/p3-relevance-classifier:latest .
    docker tag niekpas/p3-relevance-classifier:latest ghcr.io/niekvandepas/p3-relevance-classifier:latest

push:
    docker push niekpas/p3-relevance-classifier:latest
    docker push ghcr.io/niekvandepas/p3-relevance-classifier:latest

run:
    docker run --platform=linux/amd64 -it ghcr.io/niekvandepas/p3-relevance-classifier:latest /bin/bash
