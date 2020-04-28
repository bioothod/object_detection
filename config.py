import typing

class BaseConfig:
    input_size = 512
    backbone = 0
    bifpn_width = 64
    bifpn_depth = 2
    dclass = 3

PHIs = list(range(0, 8))

class DetConfig:
    def __init__(self, d=0):
        self.base_conf = BaseConfig()
        self.d = d

    @property
    def input_size(self) -> int:
        return self.base_conf.input_size + PHIs[self.d] * 128

    @property
    def bifpn_width(self) -> int:
        return int(self.base_conf.bifpn_width * 1.35 ** PHIs[self.d])

    @property
    def bifpn_depth(self) -> int:
        return self.base_conf.bifpn_depth + PHIs[self.d]

    @property
    def dclass(self) -> int:
        return self.base_conf.dclass + int(PHIs[self.d] / 3)

    @property
    def b(self) -> int:
        return self.d

class AnchorsConfig(typing.NamedTuple):
    sizes: typing.Sequence[int] = (32, 64, 128, 256, 512)
    strides: typing.Sequence[int] = (8, 16, 32, 64, 128)
    ratios: typing.Sequence[float] = (1, 2, .5)
    scales: typing.Sequence[float] = (2 ** 0, 2 ** (1 / 3.0), 2 ** (2 / 3.0))
