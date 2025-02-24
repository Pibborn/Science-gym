import numpy as np

from environments.SIRVBase import SIRVEnvironment
from environments.sirv_compartments import SIRVVitalDynamics


class SIRVOneTimeVaccination(SIRVEnvironment):
    def __init__(self, population=10_000, record_training: bool = False, recording_interval: int = 150):
        super().__init__(0, 0, population)
        self.visualize_training: bool = record_training
        self.recorded_episodes: list = []
        self.recording_interval = recording_interval  # how many steps there should be between recordings

    def apply_action(self, action: np.ndarray):
        # move a proportion of individuals into the vaccinated compartment
        self.SIRV_model.vaccinated = self.SIRV_model.susceptible * action[0]
        self.SIRV_model.susceptible -= self.SIRV_model.susceptible * action[0]

        # adjust saved initial conditions
        self.initial_susceptible = self.SIRV_model.susceptible
        self.initial_vaccinated = self.SIRV_model.vaccinated

        # do full simulation for visualisation purposes
        if self.visualize_training is True and self.episode_number % self.recording_interval == 0:
            self.render()

        # simulate a step
        self.SIRV_model.step()

    def get_reward(self):
        # calculate reward according to distance from one-to-one infection rate
        return -abs(self.SIRV_model.infected - self.initial_infected) / self.initial_infected * 100

    def is_done(self) -> bool:
        return True

    def render(self, mode="human"):
        timesteps = 10_000
        test_sim = SIRVVitalDynamics(
            transmission_rate=self.SIRV_model.transmission_rate,
            recovery_rate=self.SIRV_model.recovery_rate,
            vaccination_rate=0,
            susceptible=self.SIRV_model.susceptible,
            infected=self.SIRV_model.infected,
            recovered=self.SIRV_model.recovered,
            vaccinated=self.SIRV_model.vaccinated,
            birth_rate=self.birth_rate,
            death_rate=self.death_rate
        )

        _, _, compartments = self.run_simulation(test_sim, timesteps)

        n = self.SIRV_model.get_full_population()
        data = [[s / n * 10_000 for s, i, r, v in compartments],
                [i / n * 10_000 for s, i, r, v in compartments],
                [r / n * 10_000 for s, i, r, v in compartments],
                [v / n * 10_000 for s, i, r, v in compartments]]

        self.recorded_episodes.append(data)
