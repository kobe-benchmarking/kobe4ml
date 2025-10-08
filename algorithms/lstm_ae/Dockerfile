FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi

COPY lstm_ae ./lstm_ae

EXPOSE 48031

CMD ["poetry", "run", "api"]