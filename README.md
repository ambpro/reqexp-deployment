# Reqexp-deployment
This repository contains the Docker configuration files for the deployment of the ReqExp project.

## Reqexp Technical stack
ReqExp server needs around 8 GB of RAM. ReqExp is based on the following technology stack:
* Ubuntu 18.04
* Python 3.5+
* Flask App Server
* TensorFlow 1.14
* BERT as a Service
* Server-side Python packages: pandas, numpy, json, sklearn, spacy, bert-serving, GPUtil, subprocess

## Build and run the image

* To build the docker image using: `docker build --tag reqexp-deployment_reqexp:latest reqexp/`.
* To run the docker image using: `docker run --publish 56733:56733 --name reqexp reqexp-deployment_reqexp:latest`

## Accessing with REST endpoints
The ReqExp web server expose four REST endpoints.
* The following command sends a POST request to the endpoint (textprob) with text data (see “text” field in the JSON-payload).
`curl -i -H "Content-Type: application/json" -X POST -d '{"text":"But there are other areas that can and should be explored by the community. "}'  http://localhost:56733/textsprob`

* Add data set and retrain: 
** the following command sends a POST request to the endpoint (addtrain) with labeled text data (see “trainset” field in the JSON-payload. 
`curl -i -H "Content-Type: application/json" -X POST -d '{"trainset":  [["But there are other areas that can and should be explored by the community.  ","1"],["gui must provide data   ",  "0"]] }' http://localhost:56733/addtrain`

** The following command sends a POST request to retrain the DNN classifier model with labeled text data received in the last call of the addtrain. 
`curl -i -X POST http://localhost:56733/retrain`

* Get status of the retraining: 
** the following command sends a GET request to receive a status of the retraining process.
`curl -i -X GET http://localhost:56733/status`
