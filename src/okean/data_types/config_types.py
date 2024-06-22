from dataclasses import dataclass


@dataclass
class IndexConfig:
    ndim: int
    metric: str = "ip"
    dtype: str = "f32"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64
    multi: bool = False

    def to_dict(self):
        return self.__dict__