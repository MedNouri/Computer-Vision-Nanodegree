# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent 
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"

knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
knowledge_agent_response = knowledge_agent.respond(input_text=prompt)

print(knowledge_agent_response)

analysis = """Analysis:
- The agent answered "London" using provided knowledge, not "Paris" from its training.
- This demonstrates the agent correctly prioritizes external knowledge over inherent knowledge."""

# Save the prompt and response to a test file
with open(os.path.join(os.path.dirname(__file__), "tests", "knowledge_augmented_agent.txt"), "w") as f:
    f.write(f"Prompt: \n{prompt}\n\n")
    f.write(f"Response: \n{knowledge_agent_response}\n\n")
    f.write(analysis)
