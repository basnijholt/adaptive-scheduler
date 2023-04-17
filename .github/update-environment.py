"""Update environment.yml from pyproject.toml."""
from __future__ import annotations

import tomllib


def generate_environment_yml(
    data: dict,
    sections: tuple[str, ...] = ("all", "test", "docs"),
) -> str:
    """Generate environment.yml from pyproject.toml."""
    env_yaml = "name: adaptive-scheduler\n\n"
    env_yaml += "channels:\n- conda-forge\n\n"
    env_yaml += "dependencies:\n"
    env_yaml += "  - python\n"
    env_yaml += "  - mpi4py\n"
    env_yaml += "  - mpich\n"

    # Required deps from pyproject.toml
    for dep in data["project"]["dependencies"]:
        env_yaml += f"  - {dep}\n"

    # Optional dependencies
    for group in data["project"]["optional-dependencies"]:
        if group not in sections:
            continue
        env_yaml += f"  # optional-dependencies: {group}\n"
        for dep in data["project"]["optional-dependencies"][group]:
            env_yaml += f"  - {dep}\n"

    return env_yaml


if __name__ == "__main__":
    # Load pyproject.toml
    with open("pyproject.toml") as f:  # noqa: PTH123
        data = tomllib.loads(f.read())

    # Generate environment.yml
    environment_yml = generate_environment_yml(data, sections=("all", "test"))

    # Save environment.yml
    with open("environment.yml", "w") as f:  # noqa: PTH123
        f.write(environment_yml)

    # Generate environment.yml
    environment_yml = generate_environment_yml(data, sections=("all", "test", "docs"))

    # Save environment.yml
    with open("docs/environment.yml", "w") as f:  # noqa: PTH123
        f.write(environment_yml)
