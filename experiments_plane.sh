cd Science-gym
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Science-gym_venv/bin/activate

# SAC Plane classic SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+' '*' '-' '/' --unary_operators 'sin'  --complexity_of_constants 10 --maxsize 9

# SAC Plane classic SAC gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context classic --rl_agent SAC --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200  --binary_operators '+''*' '-' '/' --unary_operators 'cos'

# Plane classic A2C pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 10 --maxsize 9

# Plane classic A2C gplearn
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context classic --rl_agent A2C --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer gplearn --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'

# Plane sparse SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context sparse --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 10 --maxsize 9 

# Plane noise SAC pysr
/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation plane --input_cols mass gravity angle  --output_col force --context noise  --rl_agent SAC  --rl_train_steps 5_000 --rl_test_episodes 500 --equation_discoverer pysr --niterations 200 --binary_operators '+''*' '-' '/' --unary_operators 'cos'  --complexity_of_constants 10 --maxsize 9
