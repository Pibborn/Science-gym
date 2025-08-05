cd Science-gym
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.virtualenvs/Science-gym_venv/bin/activate

/home/jbrugger/.virtualenvs/Science-gym_venv/bin/python3.12  threshold_and_save.py --simulation basketball --input_cols velocity_sin_angle time g --output_col ball_y --context classic --rl_train_steps 1000 --rl_test_episodes 100 --niterations 200 --binary_operators '-' '*' '+' --unary_operators  --complexity_of_constants 5