from datetime import datetime

from services import BoardGeneratorService, BoardService, BenchMarkService
from settings import settings
from algorithms import DFS, BFS, Greedy, AStar

SOLVERS = {
    "dfs": DFS,
    "bfs": BFS,
    "greedy": Greedy,
    "a_star": AStar
}

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    initial_state = board_generator.generate()
    board_service = BoardService()

    if settings.benchmarks.active == True:
        board_benchmark_service = BenchMarkService(initial_state, settings.benchmarks.rounds)
        benchmark = board_benchmark_service.get_benchmark()
        board_benchmark_service.plot_time_comparing_graph(benchmark)
    else:
        initial_df = board_generator.dict_to_df(initial_state.regions)
        print("\nTablero inicial:")
        print(initial_df)

        solver = SOLVERS[settings.algorithm](initial_state)
        
        startTime = datetime.now()
        
        solution, cost = solver.solve()
        
        endTime = datetime.now()

        solution_df = board_generator.dict_to_df(solution.regions)
        print("\nTablero solucion:")
        print(solution_df)

        print(f"\nCosto de la solucion: {cost}")
        print(f"Tiempo de ejecucion: {(endTime - startTime).total_seconds()} segundos\n")

        if settings.visualization == True:
            # TODO: save images with steps to solution
            pass
