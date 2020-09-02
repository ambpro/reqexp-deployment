FROM ubuntu:18.04

LABEL description="Image with ReqExp platform"

WORKDIR /home/ubuntu

EXPOSE 56733

RUN apt-get update && \
    apt-get install -y \
    wget \
    unzip \
    apache2
RUN apt-get install -y \
    python3.6 \
    python3-dev \
    python3-pip

# Install pip3 packages needed
RUN pip3 install Flask==1.1.1 
RUN pip3 install --user --upgrade tensorflow==1.14
RUN pip3 install pandas numpy scikit-learn spacy GPUtil
RUN python3 -m spacy download en_core_web_sm
RUN pip3 install -U bert-serving-server bert-serving-client

COPY scripts/ /home/ubuntu/bert/
RUN wget https://github.com/ambpro/reqexp-deployment/releases/download/1.0/uncased_L-12_H-768_A-12-20200617T114030Z-001.zip 
RUN unzip -o uncased_L-12_H-768_A-12-20200617T114030Z-001.zip -d /home/ubuntu/bert/ && rm -R uncased_L-12_H-768_A-12-20200617T114030Z-001.zip
RUN wget https://github.com/ambpro/reqexp-deployment/releases/download/1.0/reqmodel_2.zip
RUN unzip -o reqmodel_2.zip -d /home/ubuntu/bert/ && rm -R reqmodel_2.zip
ADD dnn_models.zip /home/ubuntu/
RUN unzip -o dnn_models.zip && rm -R dnn_models.zip
ADD TestApp.zip /home/ubuntu/
RUN unzip -o TestApp.zip && rm -R TestApp.zip && mv TestApp /var/www/
#COPY bert_start.sh ./bert_start.sh
#RUN chmod +x ./bert_start.sh && ./bert_start.sh
COPY start.sh ./start.sh 
RUN chmod +x ./start.sh
CMD ["bash", "/home/ubuntu/start.sh"]