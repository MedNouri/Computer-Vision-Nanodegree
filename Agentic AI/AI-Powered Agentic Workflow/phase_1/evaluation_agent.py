# TODO: 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"

# TODO: 2 - Instantiate the KnowledgeAugmentedPromptAgent here
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key,persona=persona, knowledge= knowledge)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
# TODO: 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here
evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona, evaluation_criteria=evaluation_criteria, worker_agent=knowledge_agent, max_interactions=10)
# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
response = evaluation_agent.evaluate(prompt)

print(response)

explanation = """The Evaluation Agent checks if the response meets the criteria.
The criteria require only the city name, not a full sentence.
The Evaluation Agent will say if the answer is acceptable or not."""

# Save the prompt and response to a test file
with open(os.path.join(os.path.dirname(__file__), "tests", "evalution_agent.txt"), "w") as f:
    f.write(f"Prompt: \n{prompt}\n\n")
    f.write(f"Response: \n{response}\n\n")
    f.write(f"Explanation: \n{explanation}\n\n")


 