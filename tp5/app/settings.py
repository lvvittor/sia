from __future__ import annotations
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

class AdamOptimization(BaseModel):
    beta1: float
    beta2: float
    epsilon: float

class DenoisingAutoencoder(BaseModel):
    noise: float # Noise in [0, 1] to add to the inputs 
    train_iterations: int # Number of iterations to train the autoencoder for computing the percentage difference

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

    verbose: bool
    learning_rate: float
    epochs: int
    exercise: int
    optimization: str
    adam_optimization: AdamOptimization
    denoising_autoencoder: DenoisingAutoencoder 
    latent_space_points_to_add: int

    class Config:
        env_file_encoding = "utf-8"
        config_path = "tp5/config.json"
        output_path = "tp5/output"
        data_path = "tp5/data"

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
