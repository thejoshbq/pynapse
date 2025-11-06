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
            populations += f"\n - {population}"

        info = f"""Project: {self._name}\nPopulations: {populations}\nAuthors: {authors}\nDescription: {self._description}"""
        return info

if __name__ == "__main__":
    project = Project("Test Project", ["Population 1", "Population 2"], ["Author 1", "Author 2"], "This is a test project.")
    print(project)