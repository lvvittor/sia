# CMYK Color Combinator

Given a color palette, mixes different proportions of each color to create another color, similar to a target color.

![](https://github.com/lvvittor/sia/blob/master/tp2/output/demo_75p_fitness.gif)

## Running the project

Please follow the instructions on the `README.md` located in the parent folder to run the project using either `Docker` or `Poetry`.

## Configuration

You'll find a `config.json` file in this folder, which looks something like this:

```
{
    "color_palette": [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ],
    "target_color": [0, 0.9, 0.3, 1],
    "algorithm": {
        "individuals": 8,
        "selection_method": "roulette",
        "crossover_method": "uniform",
        "mutation_method": "limited",
        "mutation_rate": 0.05,
        "mutation_delta": 0.1
    },
    "visualization": {
        "display_interval": 100
    },
    "constraints": {
        "max_generations": 5000,
        "max_seconds": 60,
        "acceptable_fitness": 0.95,
        "acceptable_fitness_stagnation": 5000
    },
    "benchmarks": {
        "individuals": 8,
        "active": false,
        "rounds": 2
    }
}
```

> **Important**: the `algorithm.individuals` parameter should always be an even number.

- `color_palette`: A list of color values represented as a list of four elements. Each element represents the CYMK values (cyan, yellow, magenta, and key) for a color.
- `target_color`: A list of four elements representing the CYMK values of the target color that the program will try to generate.
- `algorithm.individuals`: An integer specifying the number of individuals in the population.
- `algortihm.selection_method`: A string specifying the selection method used by the genetic algorithm. Possible values are "elite" (selects the best individuals), "roulette" (selects individuals based on fitness and randomly), "universal" (selects individuals using a distribution based on their fitness) and "ranking" (selects individuals based on a ranking of their fitness)
- `algorithm.crossover_method`: A string specifying the crossover method used by the genetic algorithm. Possible values are "one_point" (swaps only one gene), "two_point" (swaps all genes from a given range), "anular" (swaps a number of genes from a starting point) and "uniform" (swaps all genes).
- `algorithm.mutation_method`: A string specifying the mutation method ussed by the genetic algorithm. Possible values are "limited" (an amount of genes are selected for mutation), "uniform" (all genes are selected individually for mutation), "complete" (all genes as a whole are selected for mutation).
- `algortihm.mutation_rate`: A float specifying the mutation rate for the genetic algorithm. This determines the probability of a mutation occurring with the selected genes using the mutation method during the evolution process.
- `algorithm.mutation_delta`: A float specifying the upper and lower bounds of the uniform random for the mutation process.
- `visualization.display_interval`: An integer specifying the number of iterations between displaying the best individual.
- `constraints.max_generations`: An integer specifying the maximum number of generations the algorithm will run for.
- `constraints.max_seconds`: An integer specifying the maximum number of seconds the algorithm will run for.
- `constraints.acceptable_fitness`: A float specifying the target fitness value. If an individual with this fitness or higher is found, the algorithm will stop.
- `constraints.acceptable_fitness_stagnation`: An integer specifying the number of generations with no improvement in the best fitness value before the algorithm stops.
- `benchmarks.individuals`: An integer specifying the number of individuals in the population for the benchmark.
- `benchmarks.active`: A boolean specifying whether or not to run the benchmark.
- `benchmarks.rounds`: An integer specifying the number of rounds to run the benchmark for.

# CMYK vs RGB color models

The CMYK model is better than the RGB model for mixing colors when it comes to printing or painting, because it is a subtractive color model. In the subtractive color model, when we mix colors, the result becomes darker because the pigments absorb more light.

In contrast, the RGB model is an additive color model, which is used in digital devices like computer screens and TVs. In the additive color model, when we mix colors, the result becomes brighter because the colors emit light.

Therefore, the CMYK model is more appropriate for mixing colors in printing or painting because it represents how the pigments interact with light. RGB is more appropriate for digital devices because it represents how light is emitted from a screen.
