from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initialize the FastAPI app
app = FastAPI(
    title="Text Summarization APP",
    description="This is a text summarization app using T5 model",
    version="1.0"
)

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)


# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)


# templates
templates = Jinja2Templates(directory=".")


# input schema for dialogue
class InputText(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate summary
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # Token IDs → text
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# API endpoint for summarization
@app.post("/summarize/")
async def summarize(input_text: InputText):
    summary = summarize_dialogue(input_text.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )