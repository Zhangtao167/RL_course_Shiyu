source /opt/conda/bin/activate
conda activate ideEnv

# cd assignment2/code
# plot deterministic policy
python RL_alg1.py --exp_date "2024-09-20" --policy_type "deterministic" --plot_policy 1 --is_save_gif 0 --results_path '../results'
# caculate V by closed_form method
python RL_alg1.py --exp_date "2024-09-20" --policy_type "deterministic" --plot_policy 1 --is_save_gif 0 --results_path '../results' --V_cal_method "closed_form"
# caculate V by iteration
python RL_alg1.py --exp_date "2024-09-20" --policy_type "deterministic" --plot_policy 1 --is_save_gif 0 --results_path '../results' --V_cal_method "iter"

# plot stochastic policy
python RL_alg1.py --exp_date "2024-09-20" --policy_type "stochastic" --plot_policy 1 --is_save_gif 0 --results_path '../results'
# caculate V by closed_form method
python RL_alg1.py --exp_date "2024-09-20" --policy_type "stochastic" --plot_policy 1 --is_save_gif 0 --results_path '../results' --V_cal_method "closed_form"
# caculate V by iteration
python RL_alg1.py --exp_date "2024-09-20" --policy_type "stochastic" --plot_policy 1 --is_save_gif 0 --results_path '../results' --V_cal_method "iter"