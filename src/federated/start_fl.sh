#! /bin/bash
while getopts c:s:g: flag
do
    case "${flag}" in
        c) nclients=${OPTARG};;
		s) startnumber=${OPTARG};;
		g) gpu=${OPTARG};;
    esac
done

for i in $(seq $startnumber $nclients);
do
	echo "STARTING CLIENT $i"
	docker run --name "fed-client-$i" \
	--gpus device="$gpu"\
	-d \
	--cpu-shares="500" \
	--cpus="2" \
	--tty \
	-e CLIENT_ID="$i" \
	-u $(id -u ${USER}):$(id -g ${USER}) \
	-v "$(pwd)/results/":"/results/" \
	-v "$(pwd)/data/":"/data/" \
	-v "$(pwd)/config.fl.json":"/config.fl.json" \
	-v "$(pwd)/models/model.py":"/src/model.py" \
	-v "$(pwd)/models/resnet.py":"/src/resnet.py" \
	-v "$(pwd)/models/vgg15.py":"/src/vgg15.py" \
	-v "$(pwd)/models/vgg15rnn.py":"/src/vgg15rnn.py" \
	-v "$(pwd)/client/client.py":"/src/client.py" \
	-v "$(pwd)/client/data.py":"/src/data.py" \
	-v "$(pwd)/utils/adversarial.py":"/src/adversarial.py" \
	-v "$(pwd)/utils/defense.py":"/src/defense.py" \
	-v "$(pwd)/utils/optimizers.py":"/src/optimizers.py" \
	-v "$(pwd)/utils/fs.py":"/src/fs.py" \
	fl-client
done

