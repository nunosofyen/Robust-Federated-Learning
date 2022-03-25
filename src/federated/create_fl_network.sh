#! /bin/bash
docker network rm fl-network
docker network create --driver=bridge fl-network2