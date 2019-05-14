with open("requirements.txt") as f:
    requirements = f.read().split()

template = """\
name: adaptive-scheduler

channels:
- conda-forge

dependencies:
  - python
""" + "\n".join(
    f"  - {x}" for x in requirements
)
with open("environment.yml", "w") as f:
    f.write(template + "\n")
