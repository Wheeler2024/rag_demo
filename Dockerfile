FROM langchain/langgraph-api:3.12



# set transformers cache directory to avoid permission issues inside container

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Copy entrypoint script for auto-building vector store

COPY entrypoint.sh /entrypoint.sh
# Convert line endings from CRLF to LF (for Windows compatibility)
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh



# -- Adding local package . --
ADD . /deps/rag_demo
# -- End of local package . --



# -- Installing all local dependencies --

# Install PyTorch CPU version first to avoid CUDA dependencies
RUN uv pip install --system --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

RUN for dep in /deps/*; do \
            echo "Installing $dep"; \
            if [ -d "$dep" ]; then \
                echo "Installing $dep"; \
                (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir \
                --extra-index-url https://download.pytorch.org/whl/cpu \
                -c /api/constraints.txt -e .); \
            fi; \
        done

# -- End of local dependencies install --

ENV LANGSERVE_GRAPHS='{"rag_demo": "/deps/rag_demo/src/agent.py:agent"}'







# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx



WORKDIR /deps/rag_demo

# Set entrypoint to auto-build vector store on startup
ENTRYPOINT ["/entrypoint.sh"]

# Default command to start LangGraph API server
CMD ["uvicorn", "langgraph_api.server:app", "--host", "0.0.0.0", "--port", "8000"]