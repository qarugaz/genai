#!/bin/bash

apt update
apt install vim -y
pip install huggingface_hub
pip install -r requirements.txt
huggingface-cli login --token "$HF_TOKEN"
