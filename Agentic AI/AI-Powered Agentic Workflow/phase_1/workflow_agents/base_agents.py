# TODO: 1 - import the OpenAI class from the openai library
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # TODO: 3 - Specify the model to use (gpt-3.5-turbo)
            messages=[
                # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content.strip()


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:

    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": f"You are to assume the following persona: {self.persona}. Forget any previous context or conversations. Respond to all queries as this persona only, using their knowledge, characteristics, and manner of speaking."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        # TODO: 4 - Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content.strip()


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:

    def __init__(self, openai_api_key, persona, knowledge, response_format=None):
        """Initialize the agent with provided attributes.

        Args:
            response_format: Optional structured output format, e.g.
                {"type": "json_schema", "json_schema": {...}}.
                When provided, gpt-4o-mini is used (required for json_schema support).
        """
        self.persona = persona
        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.response_format = response_format

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)

        # Use gpt-4o-mini when response_format is set (required for Structured Outputs).
        # Otherwise fall back to gpt-3.5-turbo.
        model = "gpt-4o-mini" if self.response_format else "gpt-3.5-turbo"

        kwargs = {}
        if self.response_format:
            kwargs["response_format"] = self.response_format

        response = client.chat.completions.create(
            model=model,
            messages=[
                # TODO: 2 - Construct a system message including the persona, knowledge, and final instruction.
                # TODO: 3 - Add the user's input prompt here as a user message.
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                        f"Answer the prompt based on this knowledge, not your own."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0,
            **kwargs
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(model="text-embedding-3-large", input=text, encoding_format="float")
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
        chunks, start, chunk_id = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)
            chunks.append({"chunk_id": chunk_id, "text": text[start:end], "chunk_size": end - start, "start_char": start, "end_char": end})
            start = end - self.chunk_overlap
            chunk_id += 1
        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))
        best_chunk = df.loc[df['similarity'].idxmax(), 'text']
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content


class EvaluationAgent:

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        final_response = None
        final_evaluation = None

        for i in range(self.max_interactions):  # TODO: 2 - Set loop to iterate up to the maximum number of interactions
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(input_text=prompt_to_evaluate)  # TODO: 3 - Obtain a response from the worker agent
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"  # TODO: 4 - Insert evaluation criteria here
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": eval_prompt}],  # TODO: 5 - Define the message structure for evaluation
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                final_response = response_from_worker
                final_evaluation = evaluation
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": instruction_prompt}],  # TODO: 6 - Define the message structure for correction instructions
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
                final_response = response_from_worker
                final_evaluation = evaluation

        return {
            # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": i + 1
        }


class RoutingAgent:

    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        # TODO: 1 - Define an attribute to hold the agents, call it agents
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(model="text-embedding-3-large", input=text, encoding_format="float")
        embedding = response.data[0].embedding
        return embedding

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)
            # TODO: 6 - Add logic to select the best agent based on the similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)

        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        response_text = response.choices[0].message.content  # TODO: 4 - Extract the response text

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = [step.strip() for step in response_text.split("\n") if step.strip()]

        return steps