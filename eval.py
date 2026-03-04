import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
from langchain.chat_models import init_chat_model

from src.agent import run_llm
from test_questions import QUESTIONS_WITH_ANSWERS

load_dotenv()

client = Client()

# ================================================================ #
# Step 1: Create dataset                                           #
# ================================================================ #

dataset_name = "pubquiz-eval"

if not client.has_dataset(dataset_name=dataset_name):
    # Loading question-answer example

    qa_examples = [
        {"inputs": {"question": question}, "outputs": {"answer": answer}}
        for question, answer in QUESTIONS_WITH_ANSWERS.items()
    ]
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=qa_examples)
else:
    dataset = client.read_dataset(dataset_name=dataset_name)

# ================================================================ #
# Step 2: Define target function                                   #
# ================================================================ #


def answer_question(inputs: dict) -> dict:
    return run_llm(inputs["question"], chat_memory=[])


# ================================================================ #
# Step 3: Define evaluator (with LLM judge)                        #
# ================================================================ #

llm_judge = init_chat_model("gpt-4o-mini", model_provider="openai")


def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Using a llm to to judge if the results is good (enough) in
    comparison to the ground truth data.

    """
    question = inputs["question"]
    expected = reference_outputs["answer"]
    answer = outputs.get("answer", "")

    verdict = (
        llm_judge.invoke(
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"Agent answer: {answer}\n"
            f"Is the agent answer correct? Reply only with 'correct' or 'incorrect'."
        )
        .content.strip()
        .lower()
    )
    return verdict == "correct"


# ================================================================ #
# Step 4: Run evaluation                                           #
# ================================================================ #

results = client.evaluate(
    answer_question,
    data=dataset_name,
    evaluators=[correct],
    experiment_prefix="pubquiz-agent",
    description="Evaluating pubquiz agent on example questions.",
    max_concurrency=2,
)

print(results)

# Parse results and save to json locally
eval_results = []
for result in results:
    eval_results.append(
        {
            "question": result["example"].inputs["question"],
            "expected": result["example"].outputs["answer"],
            "answer": result["run"].outputs.get("answer", ""),
            "correct": result["evaluation_results"]["results"][0].score,
        }
    )

eval_folder = "./eval_results"
os.makedirs(eval_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
eval_file_name = f"{eval_folder}/eval_results_{timestamp}.json"
with open(eval_file_name, "w") as out_file:
    json.dump(eval_results, out_file, indent=2, ensure_ascii=True)
