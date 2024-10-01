nohup python3 bbclass_gridsearch.py "ijcnn1" "osvrz" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "osvrz" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "osvrz" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_cr" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_cr" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_cr" >/dev/null 2>&1 &
wait
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_ave" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zosvrg_coord" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "szvr_g" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zospider_szo" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "zospider_coord" >/dev/null 2>&1 &
wait
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_ave" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "zosvrg_coord" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "szvr_g" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "zospider_szo" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "zospider_coord" >/dev/null 2>&1 &
wait
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_ave" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "zosvrg_coord" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "szvr_g" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "zospider_szo" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "zospider_coord" >/dev/null 2>&1 &
wait
nohup python3 bbclass_gridsearch.py "ijcnn1" "sszd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "gauss_fd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "ijcnn1" "sph_fd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "sszd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "gauss_fd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "phishing" "sph_fd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "sszd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "gauss_fd" >/dev/null 2>&1 &
nohup python3 bbclass_gridsearch.py "mushrooms" "sph_fd" >/dev/null 2>&1 &

