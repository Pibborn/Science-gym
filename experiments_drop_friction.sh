cd Science-gym
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Science-gym_venv/bin/activate

# SAC DROPFRICTION classic SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+' '*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 5

# SAC DROPFRICTION classic SAC gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200  --binary_operators '+''*' '-' '/' --unary_operators 'cos'

# DROPFRICTION classic A2C pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 5

# DROPFRICTION classic A2C gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'

# DROPFRICTION sparse SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context sparse --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 5 --success_thr 0.9

# DROPFRICTION noise SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation drop_friction --input_cols drop_length adv rec avg_vel width   --output_col y --context noise  --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 5
