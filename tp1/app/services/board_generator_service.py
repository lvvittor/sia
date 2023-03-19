from __future__ import annotations
import numpy as np

from region import Cell, Region, State

class BoardGeneratorService():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.counter = 2
        self.cells = []
        self.state = State({})
        
    def add_adjacent(self, region, i, j):
        # REGION IZQ EXISTE, APPENDEO A LA REGION IZQ
        if j > 0:
            left = self.state.regions[self.cells[i][j-1].region_id]
            if region not in left.adjacents:
                left.adjacents.append(region)
            region.adjacents.append(left)

        # REGION IZQ EXISTE, APPENDEO A LA REGION IZQ
        if i > 0:
            top = self.state.regions[self.cells[i-1][j].region_id]
            if region not in top.adjacents:
                top.adjacents.append(region)
            if top not in region.adjacents:
                region.adjacents.append(top)

    def merge_adjacent_zones(self, i, j):
        # REGION IZQ
        left = self.state.regions[self.cells[i][j-1].region_id]

        # REGION ARRIBA
        top = self.state.regions[self.cells[i-1][j].region_id]
        print("Merging regions with id ", left.id, " and ", top.id)

        # ITERAR POR TODAS LAS CELDAS DE REGION TOP Y ACTUALIZAR EL ID REGION A LEFT.ID
        for top_cell in top.cells:
            top_cell.region_id = left.id

        # CAMBIAR EL ID DE LOS ADYACENTES DE REGION TOP POR ADYACENTE CON NUMERO LEFT.ID
        for top_adjacent in top.adjacents:
            top_adjacent.adjacents.remove(top)
            if left not in top_adjacent.adjacents:
                top_adjacent.adjacents.append(left)
            

        total_cells = top.cells + left.cells
        total_adjacents = list(set(top.adjacents + left.adjacents))
        new_region = Region(left.id, left.color, total_cells, total_adjacents)
        self.state.regions.pop(top.id)
        # NUEVA REGION CON LAS CELDAS DE AMBAS + LA ACTUAL
        return new_region

    def get_region(self, new_color, i, j):
        if j==0 and i == 0:
            region = Region(1, new_color, [], [])
            self.state.regions.update({1: region})
            return region
        
        if j>0 and i>0 and new_color == self.cells[i][j-1].color and new_color == self.cells[i-1][j].color:
            return self.merge_adjacent_zones(i, j)
        
        # IZQUIERDA
        if j>0 and new_color == self.cells[i][j-1].color:
            return self.state.regions[self.cells[i][j-1].region_id]
        
        # ARRIBA
        if i>0 and new_color == self.cells[i-1][j].color:
            return self.state.regions[self.cells[i-1][j].region_id]
        
        region = Region(self.counter, new_color, [], [])
        self.state.regions.update({region.id: region})
        self.add_adjacent(region, i, j)
        self.counter += 1
        return region
        

        
    def generate(self):
        # Generate a board with n rows and n columns
        # Each cell has a random value between 0 and m-1
        # return pd.DataFrame(np.random.randint(0, self.m, size=(self.n, self.n)))

        for i in range(0, self.n):
            self.cells.append([])
            for j in range(0, self.n):
                new_color = np.random.randint(0, self.m)
                print("Creating cell with color ", new_color)
                current_region = self.get_region(new_color, i, j)
                cell = Cell(False, new_color, i, j, current_region.id)
                self.cells[i].append(cell)
                current_region.cells.append(cell)
                print("Cell in position", i, " ", j, " created in region ", current_region.id)
