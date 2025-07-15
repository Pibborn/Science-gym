import numpy as np
import pandas as pd
from sciencegym.equation import Equation

from sciencegym.simulations.simulation_interface import SimulationInterface

class ProblemInterface:

    """
    High‑level wrapper that turns a domain‑specific
    :class:`~sciencegym.simulation.base.SimulationInterface`
    into a Gym‑style environment component.

    The class does **not** inherit from :class:`gym.Env` on purpose;
    instead it exposes Gym‑compatible *spaces* so that downstream
    wrappers (or RL frameworks such as Stable‑Baselines3) can use
    them to validate actions and observations.

    Parameters
    ----------
    sim : SimulationInterface
        Concrete simulation that implements at least:
        ``step``, ``reset``, ``get_state_space`` and
        ``get_action_space``.  See :py:attr:`simulation`
        attribute below for details.

    Attributes
    ----------
    simulation : SimulationInterface
        Handle to the low‑level physics, chemistry or
        epidemiology simulator (depending on the chosen
        Science‑Gym task).  All environment dynamics are
        delegated to this object.
    observation_space : gym.spaces.Space
        Space describing the structure, bounds and dtype of
        observations returned by the simulator.  Required by
        every Gym‑compatible environment to enable sampling,
        shape introspection, and automatic sanity checks.
    action_space : gym.spaces.Space
        Space that defines the legal actions the agent may send
        to :py:meth:`simulation.step`.  Ensures that RL
        algorithms can clamp, sample, or otherwise reason about
        admissible controls.

    Examples
    --------
    >>> from sciencegym.problems import Pendulum
    >>> core_sim = Pendulum()
    >>> env = ProblemInterface(core_sim)
    >>> print(env.observation_space.shape)   # e.g. (3,)
    >>> sample_action = env.action_space.sample()
    >>> next_state = env.simulation.step(sample_action)

    Notes
    -----
    * This adapter is part of *Science‑Gym*, a research benchmark
    intended for agents that **actively design and conduct
    experiments** before attempting tasks such as symbolic
    regression on the generated data.
    * Because `ProblemInterface` defers the full Gym five‑tuple
    logic to higher‑level wrappers, it is lightweight and easy
    to subclass when adding new scientific domains.
    """

    def __init__(self, sim: SimulationInterface):
        self.simulation = sim
        self.observation_space = self.get_state_space()
        self.action_space = self.get_action_space()

    def __getattr__(self, name):
        """Delegates attribute access to instance of B if not found in A.
        Here, we are looking for attributes in the simulation even as the user is 
        interacting with the Problem.
        
        Args:
            name: Name of the attribute to be searched for in self.simulation.
            """
        if hasattr(self.simulation, name):
            return getattr(self.simulation, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


    # Methods mirrored from SimulationInterface
    def step(self, action):
        """
        Advance the simulation by **one** control cycle.

        Parameters
        ----------
        action : numpy.ndarray
            Experimental action to be performed.  Its datatype and shape must conform to
            :py:attr:`action_space`, which is delegated to the underlying
            simulation.

        Returns
        -------
        object
            Whatever the wrapped :py:meth:`Simulation.step` returns.  In most
            Gym‑style simulators this will be the 5‑tuple
            ``(next_obs, reward, terminated, truncated, info)``; in lighter
            setups it may be just ``next_obs``.  Refer to your
            :class:`Simulation` implementation for the exact signature.

        Notes
        -----
        No extra post‑processing or validation is performed—the call is passed
        straight through.
        """
        return self.simulation.step(action)

    def get_state_space(self):
        """
        Retrieve the observation/state space definition.

        Returns
        -------
        gym.spaces.Space
            A Gym‐compatible space object (e.g. :class:`gym.spaces.Box` or
            :class:`gym.spaces.Dict`) describing the dimensionality, bounds and
            dtype of observations emitted by the environment.
        """
        return self.simulation.get_state_space()

    def get_current_state(self):
        """
        Mirrored from Simulation.
        Return the most recently observed state **without** advancing the simulation.

        Useful for logging, inspection or state‑feedback control that needs
        the raw observation.

        Returns
        -------
        numpy.ndarray
            Snapshot of the current state held internally by the simulation.
        """
        return self.simulation.get_current_state()
    
    def get_action_space(self):
        """
        Retrieve the action space definition.

        Returns
        -------
        gym.spaces.Space
            Space object describing the legal range, shape and dtype of
            admissible actions.
        """
        return self.simulation.get_action_space()
    
    def get_simulation(self):
        """
        Expose the wrapped :class:`Simulation` instance.

        Returns
        -------
        Simulation
            The underlying simulation object—handy for unit tests, visualisers
            or fine‑grained diagnostic tools that need direct access.
        """
        return self.simulation

    # Interface methods
    def evaluation(self, candidate: Equation, data: pd.DataFrame):
        """
        Performs evaluation of the candidate solution to the current Problem.
        Since :class:`Problem`s might have more than one solution, this is evaluated
        against all available solutions.

        Args:
            candidate (:class:`Equation`): The current 
            data (:class:`pd.DataFrame`): _description_

        Returns:
            _type_: _description_
        """
        y_pred = candidate.evaluate(data)
        solution = self.solution()
        if type(solution) is list:
            return [np.mean((single_solution.evaluate(data) - y_pred) ** 2) for single_solution in solution ]
        else:
            y_true = self.solution().evaluate(data)
            return np.mean((y_true - y_pred) ** 2)

    def validate_context(self):
        '''
        Validates the context of the problem's simulation, see Section 4.2
        of https://dl.acm.org/doi/10.1007/978-3-031-78977-9_15.
        Currently, work in progress.
        '''
        raise NotImplementedError()

    def solution(self):
        '''
        Returns the solution or solutions for the current problem.
        '''
        raise NotImplementedError()



