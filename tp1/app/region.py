from __future__ import annotations 
from enum import Enum
from pandas import DataFrame

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

class Region():
    def __init__(
        self,
        id: int,
        color: int,
        cells: list[Cell],
        adjacents: list[Region]
    ):
        self._color = color
        self._id = id
        self._cells = cells
        self._adjacents = adjacents
        
        

class Cell():
    def __init__(
        self,
        visited: bool,
        x: int,
        y: int,
    ):
        self._visited = visited
        self._x = x
        self._y = y


class RegionFactory:
    @staticmethod
    def create(board: DataFrame):
        regions = []
        RegionFactory.build_regions(0, board, 0, 0, regions, None, None)

    @staticmethod
    def build_regions(
        board: DataFrame,
        region_number: int,
        x: int,
        y: int,
        regions: list[Region],
        current_region: Region,
        color: int,
    ):
        cell = Cell(x, y)
        if (color != board.iat[x,y]):
            current_region = Region(region_number, board.iat[x,y], [cell], [])
            regions.append(current_region)
            region_number += 1
        
        
        