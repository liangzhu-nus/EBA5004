# ISY5001
## SECTION 1 : PROJECT TITLE

### AI-Doctor - Task-oriented intelligent dialogue system.



## SECTION 2 : EXECUTIVE SUMMARY




## SECTION 3 : VIDEO


## SECTION 4 : USER GUIDE - How to start

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




## SECTION 5 : PROJECT REPORT



## SECTION 6 : MISCELLANEOUS

