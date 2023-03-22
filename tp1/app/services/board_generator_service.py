from __future__ import annotations
import numpy as np
import pandas as pd

from region import Cell, Region, State

class BoardGeneratorService():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.counter = 2
        self.cells = []
        self.state = State({}, 0, [])
        
    def add_adjacent(self, region, i, j):
        # Si existe una region a la izquierda, agrego la adyacencia con la region izquierda.
        if j > 0:
            left = self.state.regions[self.cells[i][j-1].region_id]
            if region.id not in left.adjacents and region.color != left.color:
                left.adjacents.append(region.id)
            if left.id not in region.adjacents and region.color != left.color:
                region.adjacents.append(left.id)

        # Si existe una region arriba, agrego la adyacencia con la region de arriba.
        if i > 0:
            top = self.state.regions[self.cells[i-1][j].region_id]
            if region.id not in top.adjacents and region.color != top.color:
                top.adjacents.append(region.id)
            if top.id not in region.adjacents and region.color != top.color:
                region.adjacents.append(top.id)

    def merge_adjacent_zones(self, i, j):
        left = self.state.regions[self.cells[i][j-1].region_id]
        top = self.state.regions[self.cells[i-1][j].region_id]

        # Actualizo la region de las celdas de la region superior a la celda izquierda.
        for top_cell in top.cells:
            top_cell.region_id = left.id

        # Actualizo los adyacentes de la region de arriba a la region de la izquierda
        for top_adjacent_id in top.adjacents:
            top_adjacent = self.state.regions[top_adjacent_id]
            top_adjacent.adjacents.remove(top.id)
            if left.id not in top_adjacent.adjacents:
                top_adjacent.adjacents.append(left.id)

        total_cells = top.cells + left.cells
        total_adjacents = list(set(top.adjacents + left.adjacents))

        new_region = Region(left.id, left.color, total_cells, total_adjacents)
        self.state.regions.pop(top.id)
        return new_region

    def get_region(self, new_color, i, j):
        if j==0 and i == 0:
            region = Region(1, new_color, [], [])
            self.state.regions.update({1: region})
            return region
        
        if j>0 and i>0 and new_color == self.cells[i][j-1].color and new_color == self.cells[i-1][j].color and self.cells[i][j-1].region_id != self.cells[i-1][j].region_id:
            new_region = self.merge_adjacent_zones(i, j)
            self.state.regions.update({new_region.id: new_region})
            return new_region
        
        if j>0 and new_color == self.cells[i][j-1].color:
            region = self.state.regions[self.cells[i][j-1].region_id]
            self.add_adjacent(region, i, j)
            return region
        
        if i>0 and new_color == self.cells[i-1][j].color:
            region = self.state.regions[self.cells[i-1][j].region_id]
            self.add_adjacent(region, i, j)
            return region
        
        region = Region(self.counter, new_color, [], [])
        self.state.regions.update({region.id: region})
        self.add_adjacent(region, i, j)
        self.counter += 1
        return region
        

        
    def generate(self):
        for i in range(0, self.n):
            self.cells.append([])
            for j in range(0, self.n):
                new_color = np.random.randint(0, self.m)
                current_region = self.get_region(new_color, i, j)
                cell = Cell(False, new_color, i, j, current_region.id)
                self.cells[i].append(cell)
                current_region.cells.append(cell)

        return self.state


    def dict_to_df(self, board: dict[int, Region]):
        df = pd.DataFrame(data=None, index=range(self.n), columns=range(self.n))
        for region in board.values():
            for cell in region.cells:
                df.loc[cell.x, cell.y] = region.color
        return df


    def update_state(self, new_color: int):
        self.state.update_state(new_color)
        return self.state