# TP1: Pathfinding Algorithms

The goal of this project is to design and implement different type of algorithms to find the solution to the ["Fill Zone"](http://www.mygamesworld.com/game/7682/Fill_Zone.html) game.

We'll be implementing the following algorithms:
- DFS
- BFS
- A*
- Greedy

The last two algorithms use heuristics to find the best path to the solution (see more on the report).

## Configuration

You'll find a `config.json` file in this folder, which looks something like this:

```json
{
    "board": {
        "N": 8,
        "M": 3
    },
    "algorithm": "a_star",
    "visualization": true,
    "heuristic": "composite",
    "benchmarks": {
        "active": true,
        "rounds": 5
    }
}
```
- `board.N`: size of the NxN board.
- `board.M`: amount of colors in the board.
- `algorithm`: algorithm to use to solve the game. Only runs if `benchmarks==false`.
    - Could be `dfs`, `bfs`, `greedy` or `a_star`.
- `visualization`: whether or not to create images in `tp1/output` to show the steps to the solution.
- `heuristic`: what heuristic to use with the `a_star` and `greedy` algorithms.
  - Could be `farthest_region_heuristic`, `distinct_colors`, `composite` or `cells_outside_zone`.
- `benchmarks.active`: whether or not to run benchmarks and plots for **all** the algorithms.
- `benchmarks.rounds`: amount of different boards to test each algorithm.
  - Note that each algorithm runs `rounds` time for each of the different `rounds` boards, to make a more accurate time estimation.

## Running the project

Please follow the instructions on the `README.md` located in the parent folder to run the project using either `Docker` or `Poetry`.
