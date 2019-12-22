#!/usr/bin/env bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12hvl-VWteJoJsR4vdM4-0TZyJEuZJR61' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12hvl-VWteJoJsR4vdM4-0TZyJEuZJR61" -O dataset.zip && rm -rf /tmp/cookies.txt

# -j => Don't create a directory named dataset
# -d => Destination folder
unzip dataset.zip -j -d torchseg/dataset/raw/

rm dataset.zip
