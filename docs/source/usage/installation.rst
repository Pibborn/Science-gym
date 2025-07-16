Installation Guide
==================

Welcome to **Science‑Gym** — a suite of *Gym*-compatible environments for
data‑driven scientific discovery.  This guide shows the quickest way to get the
library running on your machine, plus extra tips for GUI physics, Box2D, or
symbolic‑regression extras.

.. note::

   **Heads‑up** – Science‑Gym is still young and **pulls in
   quite a few heavy dependencies** (PyTorch, Stable‑Baselines 3, Box2D,
   OpenCV, SymPy...).  We are actively refactoring to make the *core* install
   lighter and to split GPU/GUI tooling into optional extras.  Thanks for your
   patience — stay tuned for slimmer wheels in the next minor release!

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - **Requirement**
     - **Recommended**
   * - Python
     - 3.9 – 3.12 (64‑bit)
   * - ``pip`` / ``wheel``
     - Latest (``python -m pip install -U pip wheel``)
   * - SWIG *(needed only for Box2D)*
     - 4.1 +

Installing SWIG
~~~~~~~~~~~~~~~

Some environments rely on **Box2D**, which in turn needs **SWIG** to build its
C++ bindings if a pre‑built wheel is not available for your platform.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **OS**
     - **Command**
   * - Ubuntu / Debian
     - ``sudo apt-get install swig``
   * - macOS (Homebrew)
     - ``brew install swig``
   * - Windows (Chocolatey)
     - ``choco install swig`` or download the binary installer from
       *swig.org* and add it to ``PATH``

If you do not plan to use the Box2D‑based tasks you can skip SWIG altogether.

Basic install
------------------------

.. code-block:: bash

   pip install science-gym

This installs the core library plus all other requirements. Sorry that it takes a while!

* deterministic physics environments that do **not** require Box2D
* vectorised “sandbox” tasks for regression and optimisation
* the shared simulation / problem interface

Optional extras
---------------

.. code-block:: bash

   # Reinforcement‑learning agents + PyTorch
   pip install science-gym[rl]

   # GUI & 2‑D physics (Box2D, OpenCV, Pygame)
   pip install science-gym[gui]

   # Symbolic regression + SymPy
   pip install science-gym[sym]

   # Everything (WARNING: heavy!)
   pip install science-gym[all]

Extras can be mixed, e.g. ``pip install science-gym[rl,gui]``. Note that, as of version 0.1, all of these packages are anyways installed with the basic pip install.

Bleeding‑edge / development version
-----------------------------------

.. code-block:: bash

   git clone https://github.com/Pibborn/science-gym.git
   cd science-gym
   pip install --editable .[all]   # or choose the extras you need

The ``--editable`` flag lets you hack on the source and import the package
without reinstalling.

Verifying your install
----------------------

Put the following in any python script:

.. code-block:: python

   env = Sim_Lagrange()
   input_dim, output_dim = get_env_dims(env)
   train_problem = Problem_Lagrange(env)
   print(output_dim)

Expected output::

   Observation shape: (3,)

Congratulations — Science‑Gym is ready!

Feel free to open a GitHub issue if you bump into something unexpected.  