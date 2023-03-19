from __future__ import annotations 

class Region():
    def __init__(
        self,
        id: int,
        color: int,
        cells: list[Cell],
        adjacents: list[Region]
    ):
        self.color = color
        self.id = id
        self.cells = cells
        self.adjacents = adjacents

    # Necesita un equals para comparar si dos regiones son iguales?
        

class Cell():
    def __init__(
        self,
        visited: bool,
        color: int,
        x: int,
        y: int,
        region_id: int
    ):
        self.visited = visited
        self.color = color
        self.x = x
        self.y = y
        self.region_id = region_id

class State():
    def __init__(
        self,
        regions: dict[int, Region],
        # add parameters for heuristics
    ):
        self.regions = regions



        
        
        