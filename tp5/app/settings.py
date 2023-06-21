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
    train_noise: float # Noise in [0, 1] to add to the inputs 
    data_augmentation_factor: int # Number of times augmented the training data 
    predict_rounds: int # Number of prediction rounds to do
    predict_noises: list[float] # Noise levels to use for the prediction rounds
    execute: bool # Whether to execute the denoising autoencoder or not

class MiddlePoint(BaseModel):
    execute: bool # Whether to execute the middle point or not
    first_input_index: int # Index of the first input to use
    second_input_index: int # Index of the second input to use

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
    loss_function: str
    adam_optimization: AdamOptimization
    denoising_autoencoder: DenoisingAutoencoder 
    middle_point: MiddlePoint

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
