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

    def merge_regions(
        self,
        zone,
        region
    ):
        # Updateo las celdas adyacentes
        for region_cell in region.cells:
            region_cell.color = zone.color
            region_cell.region_id = zone.id

        # Los adyacentes de la region que estoy mergeando van a ser adyacentes a la region 1
        for region_adjacent in region.adjacents:
            region_adjacent.adjacents.remove(region)
            if zone not in region_adjacent.adjacents:
                region_adjacent.adjacents.append(zone)

        total_cells = zone.cells + region.cells
        total_adjacents = list(set(zone.adjacents + region.adjacents))
        if zone in total_adjacents:
            total_adjacents.remove(zone)
        if region in total_adjacents:
            total_adjacents.remove(region)
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

        for adjacent in self.regions[1].adjacents:
            if int(adjacent.color) == int(new_color):
                zone = self.merge_regions(zone, adjacent)
                adjacents.extend(adjacent.adjacents)
        self.regions.update({1: zone})

        print("Adjacents of new zone are ")
        for adjacent in self.regions[1].adjacents:
            print("     ", adjacent.id)

        # 3° Unificar todos los adyacentes
        # zone.adjacents = list(set(adjacents))
        # zone.adjacents.remove(zone)
        return self



        
        
        