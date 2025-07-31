science-gym quickstart
==============================================

This tutorial walks you through a **single end‑to‑end run** that

1. trains an SAC agent on the *Basketball* environment (reward *context 0*, full supervision),
2. records successful experiments (that is, shots) in a CSV,
3. discovers a closed‑form equation for the ball’s trajectory via **PySR**. The ball's trajectory is an instance of the moral general notion of projectile motion.

The complete script is <download_ `run_single_experiment.py`_> and takes
about 10 minutes on a modern laptop.

.. contents::
   :local:
   :depth: 1

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 35

   * - **Requirement**
     - **Install command**
   * - Python 3.9 – 3.12
     - –
   * - Science‑Gym (core) + RL + SymPy + PySR
     - ``pip install "science-gym[rl,sym]" pysr``
   * - Stable‑Baselines 3 (SB3)
     - *included in the ``rl`` extra*
   * - Gymnasium
     - *pulled automatically by Science‑Gym*

.. note::

   Heavy libs such as PyTorch will be installed; use a virtual
   environment to keep your system site‑packages clean.

The script
----------

Save the following as **``run_single_experiment.py``**:

.. code-block:: python
   :linenos:

   import csv
   from pathlib import Path

   import numpy as np
   import pandas as pd

   from stable_baselines3.common.vec_env import DummyVecEnv
   from sciencegym.agents.StableBaselinesAgents.SACAgent import SACAgent
   from sciencegym.simulations.Simulation_Basketball import Sim_Basketball
   from sciencegym.problems.Problem_Basketball import Problem_Basketball
   from sciencegym.equation import Equation
   from pysr import PySRRegressor

   # ------------------------------------------------------------------
   TIMESTEPS       = 50_000     # 2e5 for research‑grade results
   SUCCESS_THRESH  = 80         # Basketball reward ≥ 80 marks a “good” shot
   RESULTS_DIR     = Path("quickstart_results")
   RESULTS_DIR.mkdir(exist_ok=True)
   CSV_PATH        = RESULTS_DIR / "successful_states.csv"
   # ------------------------------------------------------------------

   # 1) Environment + Problem wrapper
   sim      = Sim_Basketball(context=0, rendering=False)
   problem  = Problem_Basketball(sim)
   vec_env  = DummyVecEnv([lambda: problem])

   # 2) SAC agent
   act_dim  = int(sim.action_space.shape[0])
   obs_dim  = sim.observation_space.shape
   agent    = SACAgent(obs_dim, act_dim, policy="MlpPolicy")
   model    = agent.create_model(vec_env, verbose=0)
   model.learn(TIMESTEPS)            # training

   # 3) Evaluate & save successful episodes
   successes = []
   for _ in range(400):              # evaluation roll‑outs
       obs, _ = vec_env.reset()
       done, R = False, 0.0
       while not done:
           action, _ = model.predict(obs, deterministic=True)
           obs, reward, done, info = vec_env.step(action)
           R += reward
       if R >= SUCCESS_THRESH:
           successes.append(info[0]["terminal_observation"].flatten())

   if not successes:
       raise RuntimeError("No successful shots recorded — adjust threshold.")

   with open(CSV_PATH, "w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow(problem.variables)
       writer.writerows(successes)

   # 4) Symbolic regression (PySR)
   # Note that we pre-compute some useful variables for the final equation.
   df      = pd.read_csv(CSV_PATH)
   df["velocity_sin_angle"] = df["velocity"] * np.sin(df["angle"])
   df["g"] = 9.80665
   X       = df[["velocity_sin_angle", "time", "g"]].values
   y       = df["ball_y"].values

   model_sr = PySRRegressor(
       niterations=30,
       binary_operators=["*", "-", "+"],
       unary_operators=[],
       model_selection="best",
   ).fit(X, y, variable_names=["v*sin(θ)", "t", "g"])

   print("\nDiscovered expressions:")
   print(model_sr)

   # 5) Compare to ground‑truth
   best = model_sr.get_best().sympy_format
   gt_eq = problem.solution()        # returns sciencegym.equation.Equation
   mse = lambda yhat: np.mean((y - yhat) ** 2)

   y_pred = Equation(str(best)).evaluate(df)
   print(f"\nMSE(best) = {mse(y_pred):.4e}")
   print(f"MSE(GT)   = {mse(gt_eq.evaluate(df)):.4e}")
   print(f"Ground‑truth: {gt_eq}")
   
Running the example
-------------------

.. code-block:: bash

   python run_single_experiment.py

Console output (abridged)::

   Discovered expressions:
   1.6 * (v*sin(θ)) * t - 4.9 * t^2
   ...
   MSE(best) = 8.3e-04
   MSE(GT)   = 2.1e-16
   Ground‑truth: (v*sin(θ))*t - 4.905*t**2

You should be able to recover the equation for projectile motion, up to a constant.

Where next?
-----------

* Replace ``TIMESTEPS`` with ``200_000`` to gather more data.
* Switch ``Sim_Basketball`` → ``SIRVOneTimeVaccination`` or
  ``Sim_Lagrange`` and update the preprocessing as in
  :pyfile:`threshold_and_save.py <threshold_and_save.py>` to reproduce the
  full paper pipeline.
* Use the *multi‑context* driver script (``threshold_and_save.py``) to run the
  entire benchmark automatically.

Happy experimenting!

