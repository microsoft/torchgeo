FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.10
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "source /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

USER vscode
RUN npm install -g @devcontainers/cli@v0.30.0
