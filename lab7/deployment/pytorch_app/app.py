import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from pydantic import BaseModel


class InputText(BaseModel):
    text: str


# Load tokenizer and compiled model
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
).eval()
model = torch.compile(model)  # compiled model

app = FastAPI()


@app.post("/infer")
def infer(data: InputText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.inference_mode():
        output = model(**inputs)
    return {"embedding_shape": output.last_hidden_state.shape}
