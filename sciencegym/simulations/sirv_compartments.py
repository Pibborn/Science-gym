from abc import ABCMeta, abstractmethod


class CompartmentalModel(metaclass=ABCMeta):
    """
    Base class for constructing compartmental models.
    """
    def __init__(self, compartments: list[str]):
        """

        :param compartments: Names of the variables which are compartments, in order
        """
        # registers compartment names to allow inheriting classes with varying number of compartments
        self.compartment_names: list[str] = compartments
        self.steps = 0

    def get_full_population(self):
        return sum(self.get_compartments())

    def get_compartments(self) -> tuple[int, ...]:
        return tuple(self.__dict__[compartment_name] for compartment_name in self.compartment_names)

    def step(self):
        # calculate transitions between compartments
        transitions = self.calculate_transitions()

        # calculate any additional adjustments which don't result in transfers between compartments
        modifications = self.modify_compartments()

        # sum for total deltas
        total_changes = [t + m for t, m in zip(transitions, modifications)]

        # apply changes
        self.apply_transitions(*total_changes)

        self.steps += 1

    @abstractmethod
    def calculate_transitions(self) -> tuple[int, ...]:
        """Calculates transfers between compartments."""
        pass

    @abstractmethod
    def modify_compartments(self):
        """Applies changes to compartments which don't result in transfers from one compartment to another."""
        pass

    def apply_transitions(self, *compartment_changes):
        """Adjusts compartments according to the provided transitions"""
        for compartment, change in enumerate(compartment_changes):
            self.__dict__[self.compartment_names[compartment]] += change


class SIRCompartments(CompartmentalModel):
    """
    Basic susceptible, infected, recovered model.
    """
    def __init__(self, transmission_rate, recovery_rate, susceptible, infected, recovered=0):
        super().__init__(compartments=["susceptible", "infected", "recovered"])
        self.susceptible = susceptible
        self.infected = infected
        self.recovered = recovered

        self.transmission_rate: float = transmission_rate

        self.recovery_rate: float = recovery_rate

    def get_r_0(self):
        """Base reproduction number."""
        return self.transmission_rate/self.recovery_rate

    def calculate_new_recoveries(self):
        return self.recovery_rate * self.infected

    def calculate_new_infections(self):
        return (self.transmission_rate * self.infected * self.susceptible) / self.get_full_population()

    def calculate_transitions(self) -> tuple:
        newly_infected = self.calculate_new_infections()
        newly_recovered = self.calculate_new_recoveries()

        # deltas for each compartment
        d_susceptible = - newly_infected
        d_infected = newly_infected - newly_recovered
        d_recovered = newly_recovered

        return d_susceptible, d_infected, d_recovered

    def modify_compartments(self):
        pass


class SIRVitalDynamics(SIRCompartments):
    """
    SIR model with deaths and births.
    """
    def __init__(self, transmission_rate, recovery_rate: float,
                 susceptible, infected, recovered=0,
                 birth_rate=0, death_rate=0):
        super().__init__(transmission_rate, recovery_rate, susceptible, infected, recovered)

        self.birth_rate = birth_rate  # rate of births in population
        self.death_rate = death_rate  # percentage of deaths per step

    def get_r_0(self):
        return self.transmission_rate/(self.recovery_rate + self.death_rate)

    def modify_compartments(self):
        return self.apply_vital_dynamics()

    def apply_vital_dynamics(self):
        # get deaths for each compartment
        deaths = self.get_deaths()
        births = self.birth_rate * self.get_full_population()

        # make sure deaths are subtracted, not added
        changes = [-n for n in deaths]

        # add births to susceptible population
        changes[0] += births

        return changes

    def get_deaths(self):
        return [self.death_rate * compartment for compartment in self.get_compartments()]

    @staticmethod
    def steady(beta, gamma, death_rate=0, birth_rate=0):
        """Expected steady-state"""
        s = (gamma + birth_rate) / beta
        i = (death_rate / beta) * (beta / (gamma + death_rate) - 1)
        r = (gamma / beta) * (beta / (gamma + death_rate) - 1)
        return s, i, r


class SIRVCompartments(SIRCompartments):
    """
    SIR model expanded with a vaccinated compartment.
    """
    def __init__(self, transmission_rate, recovery_rate, vaccination_rate: float,
                 susceptible, infected, recovered=0, vaccinated=0):
        super().__init__(transmission_rate, recovery_rate, susceptible, infected, recovered)

        self.compartment_names.append("vaccinated")
        self.vaccinated = vaccinated

        self.vaccination_rate: float = vaccination_rate

    def calculate_new_vaccinations(self):
        return self.vaccination_rate * self.susceptible

    def calculate_transitions(self) -> tuple[int, ...]:
        d_susceptible, d_infected, d_recovered = super().calculate_transitions()
        newly_vaccinated = self.calculate_new_vaccinations()

        d_susceptible += - newly_vaccinated

        d_vaccinated = newly_vaccinated

        return d_susceptible, d_infected, d_recovered, d_vaccinated


class SIRVVitalDynamics(SIRVCompartments, SIRVitalDynamics):
    """
    SIRV model with deaths and births.
    """
    def __init__(self, transmission_rate, recovery_rate, vaccination_rate,
                 susceptible, infected, recovered=0, vaccinated=0, birth_rate=0, death_rate=0):
        SIRVCompartments.__init__(self, transmission_rate, recovery_rate, vaccination_rate,
                                  susceptible, infected, recovered, vaccinated)
        self.birth_rate = birth_rate
        self.death_rate = death_rate
