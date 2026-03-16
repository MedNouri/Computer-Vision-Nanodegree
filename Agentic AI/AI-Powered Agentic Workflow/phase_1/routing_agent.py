
# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent , RoutingAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona, knowledge=knowledge)

knowledge = "You know everything about Europe"
# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
europe_agnet = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona, knowledge=knowledge)

persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona, knowledge=knowledge)
routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x) # TODO: 5 - Call the Texas Agent to respond to prompts
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agnet.respond(x) # TODO: 6 - Define a function to call the Europe Agent
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        # TODO: 7 - Define a function to call the Math Agent
        "func": lambda x: math_agent.respond(x)
    }
]

routing_agent.agents = agents

# TODO: 8 - Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"

response1 = routing_agent.route("Tell me about the history of Rome, Texas")
response2 = routing_agent.route("Tell me about the history of Rome, Italy")
response3 = routing_agent.route("One story takes 2 days, and there are 20 stories")
print(response1)
print(response2)
print(response3)

# Save the prompts and responses to a test file
 
with open(os.path.join(os.path.dirname(__file__), "tests", "routing_agent.txt"), "w") as f:
    f.write("ROUTING AGENT RESPONSES\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Prompt 1: Tell me about the history of Rome, Texas\n")
    f.write(f"Response 1: \n{response1}\n\n")
    
    f.write("Prompt 2: Tell me about the history of Rome, Italy\n")
    f.write(f"Response 2: \n{response2}\n\n")
    
    f.write("Prompt 3: One story takes 2 days, and there are 20 stories\n")
    f.write(f"Response 3: \n{response3}\n")
 
 