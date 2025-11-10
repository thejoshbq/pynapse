# farm.py

from typing import List
from lot import Population

class Project:
    def __init__(
            self,
            name: str = "Brew Project",
            populations: List[Population] = None,
            authors: List[str] = None,
            description: str = None,
    ):
        self._name = name
        self._populations = populations
        self._authors = authors
        self._description = description

    # private

    # public
    def get_name(self) -> str:
        return self._name

    def set_populations(self, populations: List[Population]):
        self._populations = populations

    def get_populations(self) -> List[Population]:
        return self._populations

    def set_authors(self, authors: List[str]):
        self._authors = authors

    def get_authors(self) -> List[str]:
        return self._authors

    def set_description(self, description: str):
        self._description = description

    def get_description(self) -> str:
        return self._description

    def __str__(self):
        authors = ""
        for author in self._authors:
            authors += f"\n - {author}"

        populations = ""
        for population in self._populations:
            populations += f"\n - {population.get_name()}"

        info = f"""Project: {self._name}\nPopulations: {populations}\nAuthors: {authors}\nDescription: {self._description}"""
        return info

if __name__ == "__main__":
    from bean import Sample

    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    sample1 = Sample(
        event_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy", r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
    )
    sample2 = Sample(
        event_data=r"../../data/0 EarlyAcq/ER-L1/FOV1/ER-L1_HER-2P_HI_D1_PrL-FOV1_0_6000_191105-180825.mat",
        signal_data=r"../../data/0 EarlyAcq/ER-L1/FOV1/T2_ER-L1_HER-2P_HI-D1_PrL-FOV1_behavior-001_extractedsignals_raw.npy",
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
    )

    population1 = Population(name="Population 1", samples=[sample1, sample2])
    population2 = Population(name="Population 2", samples=[sample1, sample2])

    project = Project("Test Project", [population1, population2], ["Author 1", "Author 2"], "This is a test project.")
    print(project)