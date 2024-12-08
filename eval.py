from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate
from dotenv import load_dotenv
from prompts import PROMPT_V3

load_dotenv()

client = wrap_openai(OpenAI())


@traceable
def dialogue_agent(inputs: dict) -> dict:
    messages = [{"role": "system", "content": PROMPT_V3}, *inputs["messages"]]

    result = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.2
    )

    return {
        "message": {"role": "assistant", "content": result.choices[0].message.content}
    }


# The name or UUID of the LangSmith dataset to evaluate on.
data = "Dialogue generator dataset"

# A string to prefix the experiment name with.
experiment_prefix = "Dialogue generator experiment"


def correctness_evaluator(run, example) -> dict:
    """
    Evaluates the correctness of generated dialogue.

    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available

    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original vocabulary list from inputs
    vocabulary_input = run.inputs["inputs"]["messages"][-1]["content"]

    # Extract the model's generated dialogue
    generated_dialogue = run.outputs["message"]["content"]

    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this list of vocabulary words: {vocabulary_input}

    Evaluate the example sentences for novelty, correctness, and relevance in helping a student learn and practice with the vocabulary words. Use the following scoring rubric:
    4 = The sentences are coherent with each other, and they are sensible and illustrate the use all of the vocabulary words in a fun and interesting way.
    3 = The sentences are sensible and use all of the vocabulary words correctly.
    2 = The sentences are sensible and use some of the vocabulary words correctly.
    1 = The sentences are not sensible but they use at least one of the vocabulary words.
    0 = The sentences do not use or illustrate the vocabulary words at all.
    
    Return only the number (0-4).

    The sentences are as follows:
    {generated_dialogue}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a dialogue evaluation assistant. Respond only with a number 0-4.",
            },
            {"role": "user", "content": evaluation_prompt},
        ],
        temperature=0,
    )

    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / 4,  # Normalize to 0-1
            "explanation": f"Correctness score: {score}/4",
        }
    except ValueError:
        return {
            "key": "correctness score",
            "score": 0,
            "explanation": "Failed to parse score",
        }


def conciseness_evaluator(run, example) -> dict:
    """
    Evaluates the conciseness of generated dialogues

    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available

    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original vocabulary list from inputs
    vocabulary_input = run.inputs["inputs"]["messages"][-1]["content"]

    # Extract the model's generated dialogue
    generated_dialogue = run.outputs["message"]["content"]

    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this list of vocabulary words: {vocabulary_input}

    Evaluate the example sentences for conciseness. Do this by counting the number of complete sentences between the markers "####". Use the following scoring rubric:
    3 = The number of sentences is equal to or fewer than the number of vocabulary words.
    2 = The number of sentences is equal to the number of vocabulary words + 1.
    1 = The number of sentences is equal to the number of vocabulary words + 2.
    0 = The number of sentences is greater than the number of vocabulary words + 2.
    
    Return only the number (0-3).

    The sentences are as follows:
    ####
    {generated_dialogue}
    ####
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a dialogue evaluation assistant. Respond only with a number 0-3.",
            },
            {"role": "user", "content": evaluation_prompt},
        ],
        temperature=0,
    )

    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "conciseness score",
            "score": score / 3,  # Normalize to 0-1
            "explanation": f"Conciseness score: {score}/3",
        }
    except ValueError:
        return {
            "key": "conciseness score",
            "score": 0,
            "explanation": "Failed to parse score",
        }


# List of evaluators to score the outputs of target task
evaluators = [correctness_evaluator, conciseness_evaluator]

# Evaluate the target task
results = evaluate(
    dialogue_agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)
