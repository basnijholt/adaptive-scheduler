# Making release

This document guides a contributor through creating a release of `adaptive_scheduler`.


## Preflight checks

The following checks should be made *before* tagging the release.


### Verify that `AUTHORS.md` is up-to-date

The following command shows the number of commits per author since the last
annotated tag:
```
t=$(git describe --abbrev=0); echo Commits since $t; git shortlog -s $t..
```

## Make a release, but do not publish it yet


### Tag the release

Make an **annotated, signed** tag for the release. The tag must have the name:

```
git tag -s v<version> -m "version <version>"
```

### Build a source tarball and wheels and test it

```
rm -fr build dist
python -m build
```

### Create an empty commit for new development and tag it
```
git commit --allow-empty -m 'start development towards v<version+1>'
git tag -am 'Start development towards v<version+1>' v<version+1>-dev
```

Where `<version+1>` is `<version>` with the minor version incremented
(or major version incremented and minor and patch versions then reset to 0).
This is necessary so that the reported version for any further commits is
`<version+1>-devX` and not `<version>-devX`.


## Publish the release

### Push the tags
```
git push origin v<version> v<version+1>-dev
```

### Upload to PyPI

```
twine upload dist/*
```


## Update the [conda-forge recipe](https://github.com/conda-forge/adaptive-scheduler-feedstock)

* Fork the [feedstock repo](https://github.com/conda-forge/adaptive-scheduler-feedstock)
* Change the version number and sha256 in `recipe/meta.yaml` and commit to your fork
* Open a [Pull Request](https://github.com/conda-forge/adaptive-scheduler-feedstock/compare)
* Type `@conda-forge-admin, please rerender` as a comment
* When the tests succeed, merge
