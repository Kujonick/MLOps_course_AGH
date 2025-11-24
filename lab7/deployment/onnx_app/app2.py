import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel


class InputText(BaseModel):
    text: str


# Load tokenizer and ONNX optimized model
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
)
ort_session = ort.InferenceSession(
    "model_optimized.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app = FastAPI()


@app.post("/infer")
def infer(data: InputText):
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    output = ort_session.run(None, ort_inputs)
    return {"embedding_shape": output[0].shape}
