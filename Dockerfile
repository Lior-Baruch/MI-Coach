# MI Coach app image: FastAPI session API + Gradio practice UI.
# The model server is NOT in this image — docker-compose runs vllm/vllm-openai
# alongside it (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /mi-coach

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/ app/
COPY assets/therapist_system_prompt.txt assets/therapist_system_prompt.txt

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
