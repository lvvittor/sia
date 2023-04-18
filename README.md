# Artificial Intelligence Systems

## Running the projects with Docker


 See possible commands:

```bash
make
```

Build backend image:

```bash
make build
```


Start backend container:

```
make up
```

Run specific entrypoint of a tp:

```bash
make run-tp tp="path-to-tp"
```

Clean up containers:

```
make down
```

## Running the projects with Poetry

> Use instead of the docker image, or to install new packages.

Install:

```
curl -sSL https://install.python-poetry.org/ | python3 -
```

Uninstall:

```
curl -sSL https://install.python-poetry.org/ | python3 - --uninstall
```

Install dependencies:

```
poetry install
```

Install a new package:

```
poetry add <packageName>
```

Run script:

```
poetry run python3 $(tp)/app/main.py
```
