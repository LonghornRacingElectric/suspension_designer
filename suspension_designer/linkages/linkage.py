"""linkage.py - Linkage Builders"""

from abc import ABC, abstractmethod

class LinkageBuilder(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def design():
        pass