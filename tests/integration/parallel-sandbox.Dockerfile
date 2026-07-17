FROM alpine:3.22

RUN addgroup -g 1000 node && adduser -D -u 1000 -G node node

USER node
WORKDIR /workspace
