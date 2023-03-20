from __future__ import annotations 

class Region():
    def __init__(
        self,
        id: int,
        color: int,
        cells: list[Cell],
        adjacents: set[Region]
    ):
        self.color = color
        self.id = id
        self.cells = cells
        self.adjacents = adjacents

    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return self.id

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



        
        
        