cd Science-gym
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Science-gym_venv/bin/activate

# SAC Lagrange classic SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8   --complexity_of_constants 5

# SAC Lagrange classic SAC gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200  --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8 

# Lagrange classic A2C pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8   --complexity_of_constants 5

# Lagrange classic A2C gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200 --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8 

# Lagrange sparse SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context sparse --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8   --complexity_of_constants 5 --success_thr 0.9

# Lagrange noise SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation lagrange --input_cols distance_b1_b2 d  --context noise  --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '*' '+' --unary_operators  sqrt  --maxsize  8   --complexity_of_constants 5
