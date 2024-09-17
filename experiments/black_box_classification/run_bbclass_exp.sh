nohup python3 bbclass_gridsearch.py ijcnn1  2>&1 > ijcnn1_execution.log &
wait
nohup python3 bbclass_gridsearch.py phishing 2>&1 > phishing_execution.log &
wait
nohup python3 bbclass_gridsearch.py w8a 2>&1 > w8a_execution.log &