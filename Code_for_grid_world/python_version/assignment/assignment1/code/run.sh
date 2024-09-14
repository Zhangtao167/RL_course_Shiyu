source /opt/conda/bin/activate
conda activate ideEnv

cd assignment/assignment1/code
# plot deterministic policy
python RL_alg1.py --exp_date "2024-09-14" --policy_type "deterministic" --plot_policy 1 --is_save_gif 0
# plot trajectory of deterministic policy
python RL_alg1.py --exp_date "2024-09-14" --policy_type "deterministic" --plot_policy 0 --is_save_gif 1

# plot stochastic policy
python RL_alg1.py --exp_date "2024-09-14" --policy_type "stochastic" --plot_policy 1 --is_save_gif 0
# Plot trajectory of stochastic policy
python RL_alg1.py --exp_date "2024-09-14" --policy_type "stochastic" --plot_policy 0 --is_save_gif 1