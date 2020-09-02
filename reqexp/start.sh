export PATH=~/.local/bin:$PATH
cd /home/ubuntu
export PYTHONPATH=/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/home/ubuntu/.local/lib/python3.6/site-packages:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages
/usr/local/bin/bert-serving-start -model_dir=/home/ubuntu/bert/uncased_L-12_H-768_A-12 -tuned_model_dir=/home/ubuntu/bert/reqmodel_2 -ckpt_name=model.ckpt-998 -max_seq_len=128 -num_worker=2 &

#!/bin/sh -e
cd /var/www/TestApp
export FLASK_APP=/var/www/TestApp/main.py
export PYTHONPATH=/usr/lib/python3.6/site-packages/en_core_web_sm:/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/home/ubuntu/.local/lib/python3.6/site-packages:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python3 -m flask run --host=0.0.0.0 --port=56733