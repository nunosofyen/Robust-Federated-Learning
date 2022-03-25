#! /bin/bash
while getopts i:n:t:a:g: flag
do
    case "${flag}" in
        t) name=${OPTARG};;
        i) id=${OPTARG};;
		n) network=${OPTARG};;
		a) ip_port=${OPTARG};;
		g) gpu=${OPTARG};;
    esac
done

if [ -z "$name" ]
then
	name="client-$id"
fi

echo "client-$id"

docker run --name "$name" \
--interactive --tty \
-e SERVER_IP="$ip_port" \
-e CLIENT_ID="$id" \
-u $(id -u ${USER}):$(id -g ${USER}) \
-v "$(pwd)/results/":"/results/" \
-v "$(pwd)/data/":"/data/" \
-v "$(pwd)/models/model.py":"/src/model.py" \
-v "$(pwd)/config.fl.json":"/config.fl.json" \
-v "$(pwd)/client/client.py":"/src/client.py" \
-v "$(pwd)/models/models_rnn.py":"/src/models_rnn.py" \
-v "$(pwd)/utils/optimizers.py":"/src/optimizers.py" \
-v "$(pwd)/utils/fs.py":"/src/fs.py" \
-v "$(pwd)/models/resnet.py":"/src/resnet.py" \
-v "$(pwd)/models/vgg15.py":"/src/vgg15.py" \
-v "$(pwd)/client/data.py":"/src/data.py" \
fl-client
