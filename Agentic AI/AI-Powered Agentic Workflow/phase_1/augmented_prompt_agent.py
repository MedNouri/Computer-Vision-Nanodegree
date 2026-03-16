# TODO: 1 - Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)
# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)
# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.

analysis = """Analysis:
- The agent used geographic knowledge from its training data.
- The professor persona added 'Dear students,' and extra educational context about Paris."""

print(analysis)

# Save the prompt and response to a test file
with open(os.path.join(os.path.dirname(__file__), "tests", "augmented_prompt_agent.txt"), "w") as f:
    f.write(f"Prompt: \n{prompt}\n\n")
    f.write(f"Response: {augmented_agent_response}\n")
    f.write(f"\n{analysis}\n")