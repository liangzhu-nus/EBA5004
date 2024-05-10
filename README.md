# EBA5004
## SECTION 1 : PROJECT TITLE

### AI-Doctor - Task-oriented intelligent dialogue system.



## SECTION 2 : EXECUTIVE SUMMARY

This report presents a sophisticated AI system designed to provide efficient and accurate medical consultations by recognizing user-reported symptoms and matching them with potential diseases. The system integrates two key components: Named Entity Recognition (NER) and Named Entity Review (NERev), which work in tandem to precisely identify relevant symptoms from user inputs and validate their accuracy.

The system's architecture comprises three main stages: Data Preparation, Data Parsing and Information Extraction, and Data Applications. In the Data Preparation stage, raw text data related to diseases and symptoms is collected, cleaned, and formatted for subsequent processing. The Data Parsing and Information Extraction stage utilizes the NER module to detect disease entities and symptoms in the text, followed by the NERev module that reviews and filters out irrelevant or incorrect entities. The resulting structured information is then utilized in the Data Applications stage for various purposes, including creating a Knowledge Graph, Text Classification, and providing Auxiliary Diagnosis and Treatment support.

The system leverages advanced algorithms, including a BiLSTM-CRF model for NER and an RNN architecture with pre-trained BERT embeddings for NERev. These models are trained on carefully curated datasets, enabling the system to accurately recognize and validate medical entities.

By automating the symptom recognition and disease matching process, this AI system offers several commercial benefits. It improves medical service profitability by increasing patient throughput and reducing unnecessary referrals. Additionally, it reduces operating costs by automating tasks traditionally performed by medical professionals, while enhancing service quality, patient satisfaction, and data-driven decision-making capabilities.


## SECTION 3 : USER GUIDE - How to start

Notes:

* This installation manual is only available for operating systems: **CentOS7**.


1. Install the Anaconda scientific computing environment, including python, pip, pandas, numpy, matplotplib and other scientific computing packages.

```shell
# Install the environment package in the /root/ directory
cd /root
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sh Anaconda3-2019.07-Linux-x86_64.sh 

# Configure ~/.bashrc: 
export PATH=/root/anaconda/bin/:$PATH # Add a line
```

2. Several stand-alone tools are required to install the project.

```shell
# Install Flask
pip install Flask==1.1.1

# Install Redis database
yum install redis -y

# Install the Redis driver in Python
pip install redis

# Install gunicorn
pip install gunicorn==20.0.4

# Install supervisor
yum install supervisor -y

# Install lsof
yum install lsof -y

# Install pytorch
pip install pytorch
```

3. Installing the graph database neo4j.

```shell
# Step 1: Load the neo4j installation information into the yum search list
cd /tmp
wget http://debian.neo4j.org/neotechnology.gpg.key
rpm --import neotechnology.gpg.key
cat <<EOF>  /etc/yum.repos.d/neo4j.repo
# Write the following
[neo4j]
name=Neo4j RPM Repository
baseurl=http://yum.neo4j.org/stable
enabled=1
gpgcheck=1

# Step 2: Use the yum install command to install
yum install neo4j-3.3.5

# Step 3: Use your own configuration file
cp /data/neo4j.conf /etc/neo4j/neo4j.conf
```

4. Start the neo4j graph database and check the status.

```shell
# Start the neo4j command
neo4j start

# View Status command
neo4j status
```

5. Use scripts to generate graphs.

```shell
# Execute the script code that has been written and write the data to the neo4j database
python /data/doctor_offline/neo4j_write.py
```

6. Use scripts to train models.

```shell
# There is only one model bert-Chinese in the online part
cd /data/doctor_online/bert_server/
python train.py
```

7. Start the WeRobot service in a pending manner.

```shell
# Start the werobot service, so that users can complete the conversation with the AI doctor through the WeChat interface.
nohup python /data/wr.py &
```

8. Use supervisor to start the main logical service and its Redis service.

```shell
# supervisor configuration file brief analysis
# File path location: /data/doctor_online/main_serve/supervisor.conf

# Use gunicorn to launch the main logical service based on the Flask framework
[program:main_server]
command=gunicorn -w 1 -b 0.0.0.0:5000 app:app                    ; the program (relative uses PATH, can take args)
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=./log/main_server_out      ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=./log/main_server_error        ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)

# Start the Redis service as a session management database
[program:redis]
command=redis-server
```

```shell
# Use the supervisord command to read files in the specified directory
supervisord -c /data/doctor_online/main_serve/supervisor.conf
```

```shell
# Check the status of the started service
supervisorctl status
```

9. Start the sentence-related model service in a suspended manner.


```shell
# Start the service in a suspended manner, the code is already pre-written 
# The content in the script start.sh is gunicorn -w 1 -b 0.0.0.0:5001 app:app

nohup sh /data/doctor_online/bert_serve/start.sh &
```

10. Start and view the neo4j service (graph data query):

```shell
# The neo4j service should have been up until now
neo4j start

# To view the service startup status:
neo4j status
```

11. Take the test.

```shell
Test 1: Follow the official account (new user) and send "I have some abdominal pain recently".

Test 2: (old users) After sending "I've had some abdominal pain lately", keep sending "And there are some red dots on the left side of the abdomen".
```

Note: Datasets are located in `offline/datasets`
