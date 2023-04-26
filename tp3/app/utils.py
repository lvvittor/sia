def get_boolean_value(value: int):
    return True if value == 1 else False

def logic_and(x: list[int, int]):
    return 1 if get_boolean_value(x[0]) and get_boolean_value(x[1]) else -1
