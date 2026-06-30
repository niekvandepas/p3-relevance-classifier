default:
    @just build-vllm latest
    @just push-vllm latest

build tag="latest":
    docker buildx build --platform linux/amd64 -t niekpas/p3-relevance-classifier:{{tag}} .
    docker tag niekpas/p3-relevance-classifier:{{tag}} ghcr.io/niekvandepas/p3-relevance-classifier:{{tag}}

push tag="latest":
    docker push niekpas/p3-relevance-classifier:{{tag}}
    docker push ghcr.io/niekvandepas/p3-relevance-classifier:{{tag}}

run tag="latest":
    docker run \
        --platform=linux/amd64 \
        -e HF_TOKEN="$HF_TOKEN" \
        -it niekpas/p3-relevance-classifier:{{tag}} \
        /bin/bash

build-vllm tag="latest":
    docker buildx build \
        --platform linux/amd64 \
        -f Dockerfile.vllm \
        -t niekpas/reddit-llm-classifier:{{tag}} \
        .
    docker tag niekpas/reddit-llm-classifier:{{tag}} ghcr.io/niekvandepas/reddit-llm-classifier:{{tag}}

push-vllm tag="latest":
    docker push niekpas/reddit-llm-classifier:{{tag}}
    docker push ghcr.io/niekvandepas/reddit-llm-classifier:{{tag}}

run-vllm tag="latest":
    docker run \
        --platform=linux/amd64 \
        -e HF_TOKEN="$HF_TOKEN" \
        -it niekpas/reddit-llm-classifier:{{tag}} \
        /bin/bash
