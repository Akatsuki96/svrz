nohup python3 ijcnn1.py 2>&1 > ijcnn1_execution.log &
wait
nohup python3 phishing.py 2>&1 > phishing_execution.log &
wait
nohup python3 w8a.py 2>&1 > w8a_execution.log &