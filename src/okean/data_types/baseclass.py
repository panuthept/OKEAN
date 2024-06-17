from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BaseDataType:
    def from_dict(self, dict_data: Dict[str, Any]) -> "BaseDataType":
        for key, value in dict_data.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__