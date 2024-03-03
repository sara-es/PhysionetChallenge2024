from dataclasses import dataclass, field

from reconstruction.Point import Point


@dataclass(frozen=True)
class Rectangle:
    top_left: Point = field()
    bottom_right: Point = field()

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x

