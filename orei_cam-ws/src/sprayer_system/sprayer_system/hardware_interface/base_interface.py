from abc import ABC, abstractmethod
from typing import List

class BaseHardwareInterface(ABC):
    @abstractmethod
    def set_nozzle_state(self, state: List[bool]): pass
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def disconnect(self): pass
    @abstractmethod
    def is_connected(self) -> bool: pass