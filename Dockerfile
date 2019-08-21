FROM ubuntu:16.04
LABEL MAINTAINER=sminot@fredhutch.org

# Install prerequisites
RUN apt update && \
    apt-get install -y wget unzip python2.7 python-pip hmmer
RUN pip install tensorflow>=1.4.0 setproctitle

# Add repository data
RUN mkdir /usr/local/rgn
ADD ./ /usr/local/rgn/

WORKDIR /usr/local/rgn/model/
RUN python protling.py -h
