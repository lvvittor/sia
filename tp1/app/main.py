from services import BoardGeneratorService
from settings import settings

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.N, settings.M)
    board = board_generator.generate()
    print(board)
