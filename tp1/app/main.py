from region import State
from services import BoardGeneratorService, BoardService, BenchMarkService
from settings import settings
from algorithms import DFS, BFS
import pandas as pd


def solve_algorithm(state: State):
    if settings.benchmarks.active == True:
        return
    match settings.algorithm:
        case "dfs":
            dfs_solver = DFS(state)

            initial_df = board_generator.dict_to_df(state.regions)
            #print(initial_df)

            solution, cost = dfs_solver.solve()
            solution_df = board_generator.dict_to_df(solution.regions)
            #print("Tablero solucion:")
            #print(solution_df)

            print(f"Costo de la solucion: {cost}")

        case "bfs":
            bfs_solver = BFS(state)

            initial_df = board_generator.dict_to_df(state.regions)
            print(initial_df)

            solution, cost = bfs_solver.solve()
            solution_df = board_generator.dict_to_df(solution.regions)
            print("Tablero solucion:")
            print(solution_df)

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    initial_state = board_generator.generate()
    board_service = BoardService()

    if settings.benchmarks.active == True:
        board_benchmark_service = BenchMarkService(initial_state, settings.benchmarks.rounds)
        benchmark = board_benchmark_service.get_benchmark()
        board_benchmark_service.plot_time_comparing_graph(benchmark)

    df = board_generator.dict_to_df(initial_state.regions)
    #print(df)
    solve_algorithm(initial_state)

    # i = 0
    # while True:
    #     df = board_generator.dict_to_df(board.regions)
    #     print(df)

    #     # This prints will not work because i removed those methods as there is no DataFrame inside BoardService anymore.
    #     # TODO: remove/change prints
    #     # with pd.option_context('display.max_rows', settings.board.N, 'display.max_columns', settings.board.N):
    #         # print(board_service.get_board())
        
    #     # print(f"Board [colors={board_service.get_board_colors()}, len(colors)={board_service.get_board_color_count()}]")
        
    #     board_service.set_colored_board(df, f"test{i}.png")
    #     i += 1

    #     new_color = input("What color do you want to change: ")

    #     board = board_generator.update_state(int(new_color))
  


