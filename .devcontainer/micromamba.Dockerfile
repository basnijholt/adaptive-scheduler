ARG BASE_IMAGE=git-515d637-jammy
FROM ghcr.io/mamba-org/micromamba-devcontainer:$BASE_IMAGE

# Ensure that all users have read-write access to all files created in the subsequent commands.
ARG DOCKERFILE_UMASK=0000

# Install the fix-permissions script
ADD --chmod=755 https://raw.githubusercontent.com/jupyter/docker-stacks/a5b40a6f1117bd675565b3673e063125dd74eac3/images/docker-stacks-foundation/fix-permissions /usr/local/bin/fix-permissions

# Create a fixed group for /opt/conda in case the user GID changes
RUN sudo groupadd --gid 46328 mamba-admin && sudo usermod -aG mamba-admin "${MAMBA_USER}"

# Install the Conda packages.
ARG ENVIRONMENT_YAML=environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER ${ENVIRONMENT_YAML} /tmp/environment.yml
RUN echo "use_lockfiles: false" >> ~/.mambarc && \
    micromamba config append channels conda-forge && \
    micromamba install --yes --name base --file /tmp/environment.yml && \
    micromamba clean --all --force-pkgs-dirs --yes && \
    sudo -E "NB_GID=mamba-admin" fix-permissions "${MAMBA_ROOT_PREFIX}"

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Create and set the workspace folder
ARG CONTAINER_WORKSPACE_FOLDER=/workspaces/default-workspace-folder
RUN mkdir -p "${CONTAINER_WORKSPACE_FOLDER}"
WORKDIR "${CONTAINER_WORKSPACE_FOLDER}"

# On WSL the repo is mounted with permissions 1000:1000, but the mambauser inside the container has a different UID.
# So we change the UID of the mambauser inside the container to match the UID of the user on the host.
USER root
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupmod --gid $USER_GID $MAMBA_USER \
    && usermod --uid $USER_UID --gid $USER_GID $MAMBA_USER \
    && chown -R $USER_UID:$USER_GID /home/$MAMBA_USER
USER $MAMBA_USER
