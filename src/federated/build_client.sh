#! /bin/bash
docker image rmi -f fl-client
docker build -t fl-client -f "client/Dockerfile" .