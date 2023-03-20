# Implementacion de heuristicas
# `board` es el tablero con los colores de las celdas
# `N` es el tamaño del tablero
# `visited` es una matriz del mismo tamaño que el tablero con 0 en las celdas que no pertenecen a la zona y 1 en las que si

def get_number_of_islands(board, visited, N):
  """Devuelve la cantidad de regiones (islas) fuera de la zona y los colores que las componen"""
  _visited = visited.copy() # descartamos las casillas que ya estan en la zona
  count = 0
  colors = set()
  for i in range(N):
    for j in range(N):
      if _visited[i][j] == 0:
        count += 1
        visit_neighbors(board, visited, N, i, j, board[i][j])
        colors.add(board[i][j])

  return count, colors


def visit_neighbors(board, visited, N, i, j, color):
  """Visita los vecinos de la casilla (i, j) que sean del mismo color"""
  if i < 0 or i >= N or j < 0 or j >= N:
    return
  if visited[i][j] == 1:
    return
  if board[i][j] != color:
    return

  visited[i][j] = 1

  visit_neighbors(board, visited, N, i-1, j, color)
  visit_neighbors(board, visited, N, i+1, j, color)
  visit_neighbors(board, visited, N, i, j-1, color)
  visit_neighbors(board, visited, N, i, j+1, color)