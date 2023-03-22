from __future__ import annotations
import copy

class Region:
    def __init__(
        self,
        id: int,
        color: int,
        cells: list[Cell],
        adjacents: set[int]
    ):
        self.color = color
        self.id = id
        self.cells = cells
        self.adjacents = adjacents

    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return self.id
        

class Cell:
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

class State:
    def __init__(
        self,
        regions: dict[int, Region],
        # add parameters for heuristics
    ):  
        self.regions = regions

    def merge_regions(
        self,
        zone,
        region
    ):
        # Updateo las celdas adyacentes
        for region_cell in region.cells:
            region_cell.region_id = zone.id

        # Los adyacentes de la region que estoy mergeando van a ser adyacentes a la region 1
        for region_adjacent_id in region.adjacents:
            region_adjacent = self.regions[region_adjacent_id]
            region_adjacent.adjacents.remove(region.id)
            if zone.id not in region_adjacent.adjacents and region_adjacent_id != zone.id:
                region_adjacent.adjacents.append(zone.id)

        total_cells = zone.cells + region.cells
        total_adjacents = list(set(zone.adjacents + region.adjacents))

        total_adjacents.remove(zone.id)
        if region.id in total_adjacents:
            total_adjacents.remove(region.id)
        new_region = Region(zone.id, zone.color, total_cells, total_adjacents)
        self.regions.pop(region.id)
        return new_region


    def update_state(
        self,
        new_color: int
    ):
        zone = self.regions[1]

        # 1° Updatear color de tu zona
        zone.color = new_color

        # 2° Adyacentes de la zona 1 del nuevo color mergearlos a la region 1
        adjacents = []
        regions_added = 0
        zone_adjacents_copy = zone.adjacents.copy()
        for adjacent_id in zone_adjacents_copy:
            adjacent = self.regions[adjacent_id]
            if int(adjacent.color) == int(new_color):
                zone = self.merge_regions(zone, adjacent)
                regions_added += 1
                adjacents.extend(adjacent.adjacents)
        self.regions.update({1: zone}) 

        # 3° Unificar todos los adyacentes
        # zone.adjacents = list(set(adjacents))
        # zone.adjacents.remove(zone)
        return self, regions_added


    def copy(self):
        return copy.deepcopy(self)

        
        
        