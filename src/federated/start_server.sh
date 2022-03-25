#! /bin/bash
while getopts a:n: flag
do
    case "${flag}" in
        a) ip=${OPTARG};;
        n) network=${OPTARG};;
    esac
done

if [ -z "$ip" ]
then
    echo "NO IP ADDRESS SPECIFIED, USING DEFAULT IP"
fi

echo "STARTING SERVER"
docker run --name fed-server \
--cpu-shares="500" \
--cpus="2" \
--gpus device=1 \
-l DEBUG \
-v "$(pwd)/results/":"/src/results/" \
-v "$(pwd)/config.fl.json":"/config.fl.json" \
-v "$(pwd)/server/server.py":"/src/server.py" \
-v "$(pwd)/models/model.py":"/src/model.py" \
-v "$(pwd)/models/resnet.py":"/src/resnet.py" \
-v "$(pwd)/models/vgg15.py":"/src/vgg15.py" \
-v "$(pwd)/models/vgg15rnn.py":"/src/vgg15rnn.py" \
-v "$(pwd)/models/models_res.py":"/src/models_res.py" \
-v "$(pwd)/utils/optimizers.py":"/src/optimizers.py" \
-v "$(pwd)/utils/fs.py":"/src/fs.py" \
-d --tty --ip "$ip" fl-server

