# population.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Encapsulates a population of samples.

This module defines the Population class, which aggregates a collection of
samples and provides computed properties related to the population such as
the total number of samples, neurons, and events. Population instances can
also include metadata such as name and description.

Classes:
    Population: Represents a group of Sample objects with additional
                computed properties and metadata.
"""

from typing import Any, List, Optional, Union
from pynapse.core.sample import Sample
from pynapse.core.mixins import TensorConfigMixin


class Population(TensorConfigMixin):
    def __init__(
        self,
        samples: List[Sample],
        name: str | None = None,
        description: str = None,
        default_event_id: Optional[Union[int, List[int]]] = None,
        default_pre_event: Optional[float] = None,
        default_post_event: Optional[float] = None,
        default_buffer_ms: int = 0,
        default_min_trials: int = 1,
    ):
        self._samples = samples
        self._description = description
        self._name = name or self.__class__.__name__

        self._init_tensor_config(
            default_event_id=default_event_id,
            default_pre_event=default_pre_event,
            default_post_event=default_post_event,
            default_buffer_ms=default_buffer_ms,
            default_min_trials=default_min_trials,
        )

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

    def create_tensor(
        self,
        event_id: Union[int, List[int]],
        pre_event: float,
        post_event: float,
        buffer_ms: int,
        min_trials: int,
        trace_preprocess: Any,
        window_preprocess: Any,
    ) -> Any:
        from pynapse.analysis.peri_event import PopulationEventTensor
        return PopulationEventTensor(
            population=self,
            event_id=event_id,
            pre_event=pre_event,
            post_event=post_event,
            buffer_ms=buffer_ms,
            min_trials=min_trials,
            trace_preprocess=trace_preprocess,
            window_preprocess=window_preprocess,
        )

    def __str__(self):
        name = f"Name: {self.name}"
        desc = f"Description: {self.description}"
        samples = "Samples:"
        for sample in self._samples:
            samples += f"\n - {sample.name}"
        return f"{name}\n{desc}\n{samples}"

if __name__ == "__main__":
    from pynapse.config.events import LEGACY_HER

    s1 = Sample(
        name="Sample 1",
        event_data=[r"./data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat",
                    r"./data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"./data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy",
                     r"./data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=LEGACY_HER,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"./data/empty.mat"
    )

    s2 = Sample(
        name="Sample 2",
        event_data=[r"./data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat",
                    r"./data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"./data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy",
                     r"./data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=LEGACY_HER,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"./data/empty.mat"
    )

    population = Population(name="Population 1", samples=[s1, s2])
    print(population)
    print(population.get_tensor([22,222], 10, 11.6))
