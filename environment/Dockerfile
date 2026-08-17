# hash:sha256:8d57ff955cbbe00d585be0df3d10c8433950be6f8edf644d448661374db73f54
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CLAUDE_CODE_USE_BEDROCK=1
ENV AWS_REGION=us-west-1

RUN conda install -y --channel=conda-forge \
        graphviz==8.1.0 \
        pygraphviz==1.11 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    adjusttext==1.4.0 \
    allensdk==2.16.2 \
    anndata==0.11.4 \
    cairosvg==2.9.0 \
    descartes==1.1.0 \
    igraph==1.0.0 \
    ipfx==2.0.0 \
    kaleido==1.3.0 \
    leidenalg==0.12.0 \
    matplotlib==3.10.9 \
    matplotlib-scalebar==0.9.0 \
    morph-utils==1.5.9 \
    neuron-morphology==1.2.2 \
    numpy==1.23.5 \
    openpyxl==3.1.5 \
    pandas==1.5.3 \
    plotly==6.8.0 \
    pyarrow==25.0.1 \
    pypdf2==3.0.1 \
    scikit-posthocs==0.14.0 \
    scipy==1.10.1 \
    seaborn==0.13.2 \
    statannotations==0.7.2 \
    statsmodels==0.14.6 \
    svgpathtools==1.7.2 \
    umap-learn==0.5.12

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension anthropic.claude-code@2.1.81 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python@2025.14.0 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter@2025.4.1 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r@2.8.6 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev@3.35.1

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
