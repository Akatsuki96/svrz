nohup python3 bbclass_gridsearch.py "ijcnn1" "osvrz" &
nohup python3 bbclass_gridsearch.py "phishing" "osvrz" &
nohup python3 bbclass_gridsearch.py "mushrooms" "osvrz" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_cr" &
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_cr" &
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_cr" &
wait
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_ave" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_coord" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "szvr_g" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zospider_szo" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zospider_coord" &
wait
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_ave" &
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_coord" &
nohup python3 bbclass_gridsearch.py "phishing" "szvr_g" &
nohup python3 bbclass_gridsearch.py "phishing" "zospider_szo" &
nohup python3 bbclass_gridsearch.py "phishing" "zospider_coord" &
wait
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_ave" &
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_coord" &
nohup python3 bbclass_gridsearch.py "mushrooms" "szvr_g" &
nohup python3 bbclass_gridsearch.py "mushrooms" "zospider_szo" &
nohup python3 bbclass_gridsearch.py "mushrooms" "zospider_coord" &
wait
nohup python3 bbclass_gridsearch.py "ijcnn1" "sszd" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "gauss_fd" &
nohup python3 bbclass_gridsearch.py "ijcnn1" "sph_fd" &
nohup python3 bbclass_gridsearch.py "phishing" "sszd" &
nohup python3 bbclass_gridsearch.py "phishing" "gauss_fd" &
nohup python3 bbclass_gridsearch.py "phishing" "sph_fd" &
nohup python3 bbclass_gridsearch.py "mushrooms" "sszd" &
nohup python3 bbclass_gridsearch.py "mushrooms" "gauss_fd" &
nohup python3 bbclass_gridsearch.py "mushrooms" "sph_fd" &

