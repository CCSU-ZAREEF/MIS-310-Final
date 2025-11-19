# AI Resume Analyzer – Resume-to-Job Match
# Group Project – MIS / Intro to Programming
# Team Members: (Add your names)

import os
import openai
import gradio as gr
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# --------------- Setup ---------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# --------------- Functions ---------------
def get_embedding(text):
    """Return the embedding vector for a piece of text using OpenAI Embeddings API"""
    # FIX 1: Indent the API call and the return statement inside the function
    response = openai.embeddings.create(model="text-embedding-3-small",input=text)
    return response.data[0].embedding


def rank_resumes(job_desc, resumes):
    """Compute similarity between job description and resumes"""
    job_emb = get_embedding(job_desc)
    results = []

    for name, text in resumes.items():
        # FIX 2: Indent the body of the for loop
        emb = get_embedding(text)
        score = cosine_similarity([job_emb], [emb])[0][0]
        results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def analyze(job_text, *resume_texts):
    """Main function for Gradio app"""
    resumes = {}
    for i, txt in enumerate(resume_texts, 1):
        if txt.strip():
            resumes[f"Resume {i}"] = txt.strip()

    ranked = rank_resumes(job_text, resumes)
    output = "Ranked Resumes (Best → Worst):\n\n"
    for i, (name, score) in enumerate(ranked, start=1):
        output += f"{i}. {name} — Similarity: {score:.3f}\n"
    return output


# --------------- Load Sample Data ---------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


sample_resumes = {
    "resume1": read_file("sample_data/resume1.txt"),
    "resume2": read_file("sample_data/resume2.txt"),
    "resume3": read_file("sample_data/resume3.txt"),
    "resume4": read_file("sample_data/resume4.txt"),
    "resume5": read_file("sample_data/resume5.txt"),
}
job_desc_default = read_file("sample_data/job.txt")

# --------------- Gradio Interface ---------------
with gr.Blocks(title="AI Resume Analyzer") as demo:
    gr.Markdown("# 🧠 AI Resume Analyzer – Resume-to-Job Match")
    gr.Markdown("This app ranks multiple resumes against a job posting using AI embeddings.")

    job_box = gr.Textbox(label="Job Description", value=job_desc_default, lines=6)
    gr.Markdown("### Paste or edit up to 5 resumes below:")
    r1 = gr.Textbox(label="Resume 1", value=sample_resumes["resume1"], lines=8)
    r2 = gr.Textbox(label="Resume 2", value=sample_resumes["resume2"], lines=8)
    r3 = gr.Textbox(label="Resume 3", value=sample_resumes["resume3"], lines=8)
    r4 = gr.Textbox(label="Resume 4", value=sample_resumes["resume4"], lines=8)
    r5 = gr.Textbox(label="Resume 5", value=sample_resumes["resume5"], lines=8)

    go = gr.Button("Analyze")
    output_box = gr.Textbox(label="Ranked Results", lines=10)
    go.click(analyze, inputs=[job_box, r1, r2, r3, r4, r5], outputs=output_box)

if __name__ == "__main__":
    demo.launch(share=True)

