This is a docker image to run the the project inside a docker image in order to train a centralized model for comparison to the FL model:

1. cd into:
'''
src/
'''

2. build the image :
'''
docker build -t fl-centralized .
'''

3. run the container and mount inside it the whole project:
'''
docker run --gpus all -v "$(pwd)/":"/workspace" --rm -it --entrypoint bash fl-centralized
'''