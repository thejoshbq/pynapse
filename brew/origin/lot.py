# lot.py

from typing import List
from bean import Sample

class Population:
    def __init__(
        self,
        samples: List[Sample],
        name: str = "Brew Population",
        description: str = None,
    ):
        self._samples = samples
        self._description = description
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def num_samples(self):
        return len(self._samples)

    @property
    def num_neurons(self):
        n = 0
        for sample in self._samples:
            n += sample.num_neurons
        return n

    @property
    def num_events(self):
        n = 0
        for sample in self._samples:
            n += sample.num_events
        return n

    def get_samples(self):
        return self._samples

    def __str__(self):
        name = f"Name: {self.name}"
        desc = f"Description: {self.description}"
        samples = "Samples:"
        for sample in self._samples:
            samples += f"\n - {sample.name}"
        return f"{name}\n{desc}\n{samples}"

if __name__ == "__main__":
    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    s1 = Sample(
        name="Sample 1",
        event_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat",
                    r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy",
                     r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"../../data/empty.mat"
    )

    s2 = Sample(
        name="Sample 2",
        event_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat",
                    r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy",
                     r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"../../data/empty.mat"
    )

    population = Population(name="Population 1", samples=[s1, s2])
    print(population)
