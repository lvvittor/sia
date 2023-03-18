from pydantic import BaseSettings

class Settings(BaseSettings):
    M: int = 10 # number of colors
    N: int = 10  # board of size NxN

settings = Settings()
