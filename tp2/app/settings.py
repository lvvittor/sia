from typing import Any
from pydantic import BaseSettings, BaseModel
from pathlib import Path
import json

def json_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    """
    A simple settings source that loads variables from a JSON file
    at the project's root.
    """
    encoding = settings.__config__.env_file_encoding
    config_path = settings.__config__.config_path
    return json.loads(Path(config_path).read_text(encoding))


class AlgorithmSettings(BaseModel):
    individuals: int
    selection_method: str
    crossover_method: str
    mutation_method: str
    mutation_rate: float
    mutation_delta: float

class VisualizationSettings(BaseModel):
    display_interval: int
    
class ConstraintsSettings(BaseModel):
    max_generations: int
    max_seconds: int
    acceptable_fitness: float
    acceptable_fitness_stagnation: int

class Benchmarks(BaseModel):
    individuals: int
    active: bool
    rounds: int

class Settings(BaseSettings):
    """
    Settings for the application.

    Settings are loaded from the following sources, in order:
    1. Environment variables
    2. JSON file at the project's root
    3. Secret environment variables

    For more information about parsing json files, see:

    https://jsontopydantic.com/
    """
    color_palette: list[list[float]]
    target_color: list[float]
    algorithm: AlgorithmSettings
    visualization: VisualizationSettings
    constraints: ConstraintsSettings
    benchmarks: Benchmarks

    class Config:
        env_file_encoding = 'utf-8'
        config_path = 'tp2/config.json'
        output_path = 'tp2/output'

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )


settings = Settings()
