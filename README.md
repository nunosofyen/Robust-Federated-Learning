# Adjust the FL configuration:
In order to change the FL configuration and the model parameters, open the config JSON file ```config.fl.json``` where you can overwrite the existing configuration.  


# Launch FL using Docker

1. cd into src/federated

2. build the Docker images:
```
sh build_server.sh && sh build_client.sh
```
3. run the server container in a new terminal:
```
sh start_server.sh
```
4. open another terminal, and run the client containers, and specify the number of clients (get num_clients from the config file):
```
sh start_fl -c 3
```

5. check if the client containers are running by the commend:
```
docker ps
```

6. choose one docker container ID of the client containers, and attach to its terminal output to check the training status:
```
docker attach <container-id>
```

7. when the training is done, the model is saved under "src/federated/results/"

