cd Science-gym
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Science-gym_venv/bin/activate

# SAC Basketball classic SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context classic --rl_agent SAC --rl_train_steps 50_000 --rl_test_episodes 5_000 --equation_discoverer pysr --niterations 200 --binary_operators '-' '*' '+' --unary_operators  --complexity_of_constants 5

# SAC Basketball classic SAC gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context classic --rl_agent SAC --rl_train_steps 50_000 --rl_test_episodes 5_000 --niterations 200 --equation_discoverer gplearn --binary_operators '-' '*' '+' --unary_operators

# Basketball classic A2C pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context classic --rl_agent A2C --rl_train_steps 50_000 --rl_test_episodes 5_000 --equation_discoverer pysr --niterations 200 --binary_operators '-' '*' '+' --unary_operators  --complexity_of_constants 5

# Basketball classic A2C gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context classic --rl_agent A2C --rl_train_steps 50_000 --rl_test_episodes 5_000 --equation_discoverer gplearn --niterations 200 --binary_operators '-' '*' '+' --unary_operators

# Basketball sparse SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context sparse --rl_agent SAC  --rl_train_steps 50_000 --rl_test_episodes 5_000 --equation_discoverer pysr --niterations 200 --binary_operators '-' '*' '+' --unary_operators  --complexity_of_constants 5 --success_thr 0.9

# Basketball noise SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context noise  --rl_agent SAC  --rl_train_steps 50_000 --rl_test_episodes 5_000 --equation_discoverer pysr --niterations 200 --binary_operators '-' '*' '+' --unary_operators  --complexity_of_constants 5