# agentic_workflow.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from phase_1.workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os

from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv(dotenv_path=Path(__file__).parent.parent / 'phase_1' / '.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
file_path = os.path.join(os.path.dirname(__file__), 'Product-Spec-Email-Router.txt')
with open(file_path, 'r') as file:
    product_spec = file.read()
import json
# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_stories",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "stories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "persona": {"type": "string", "description": "The type of user only, e.g. 'Customer Support Representative'. Do NOT include 'As a'."},
                                "action":  {"type": "string", "description": "The action or feature the user wants. Do NOT include 'I want'."},
                                "outcome": {"type": "string", "description": "The benefit or value. Do NOT include 'so that'."}
                            },
                            "required": ["persona", "action", "outcome"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["stories"],
                "additionalProperties": False
            }
        }
    }
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = (
    "The answer should be user stories that follow this structure:\n"
    "As a [type of user], I want [an action or feature] so that [benefit/value].\n"
    "Each story should be clear, concise, and focused on a specific user need or functionality."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups. "
    f"The product being planned is the Email Router described in this spec: {product_spec}"
)
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_features",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "feature_name":      {"type": "string"},
                                "description":       {"type": "string"},
                                "key_functionality": {"type": "string"},
                                "user_benefit":      {"type": "string"}
                            },
                            "required": ["feature_name", "description", "key_functionality", "user_benefit"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["features"],
                "additionalProperties": False
            }
        }
    }
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story. "
    f"The product being planned is the Email Router described in this spec: {product_spec}"
)
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "engineering_tasks",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id":             {"type": "string"},
                                "task_title":          {"type": "string"},
                                "related_user_story":  {"type": "string", "description": "Full 'As a ... I want ... so that ...' sentence specific to the Email Router. No placeholders."},
                                "description":         {"type": "string"},
                                "acceptance_criteria": {"type": "string"},
                                "estimated_effort":    {"type": "string"},
                                "dependencies":        {"type": "string"}
                            },
                            "required": ["task_id", "task_title", "related_user_story", "description", "acceptance_criteria", "estimated_effort", "dependencies"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["tasks"],
                "additionalProperties": False
            }
        }
    }
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work\n"
        "Related User Story: Reference to the parent user story\n"
        "Description: Detailed explanation of the technical work required\n"
        "Acceptance Criteria: Specific requirements that must be met for completion\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first"
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent
def product_manager_support_function(query):
    raw = product_manager_knowledge_agent.respond(input_text=query)
    data = json.loads(raw)
    lines = [
        f"As a {s['persona']}, I want {s['action']} so that {s['outcome']}."
        for s in data["stories"]
    ]
    return "\n".join(lines)

def program_manager_support_function(query):
    raw = program_manager_knowledge_agent.respond(input_text=query)
    data = json.loads(raw)
    blocks = [
        f"Feature Name: {f['feature_name']}\n"
        f"Description: {f['description']}\n"
        f"Key Functionality: {f['key_functionality']}\n"
        f"User Benefit: {f['user_benefit']}"
        for f in data["features"]
    ]
    return "\n\n".join(blocks)

def development_engineer_support_function(query):
    raw = development_engineer_knowledge_agent.respond(input_text=query)
    data = json.loads(raw)
    blocks = [
        f"Task ID: {t['task_id']}\n"
        f"Task Title: {t['task_title']}\n"
        f"Related User Story: {t['related_user_story']}\n"
        f"Description: {t['description']}\n"
        f"Acceptance Criteria: {t['acceptance_criteria']}\n"
        f"Estimated Effort: {t['estimated_effort']}\n"
        f"Dependencies: {t['dependencies']}"
        for t in data["tasks"]
    ]
    return "\n\n".join(blocks)

# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
        {
            "name": "Product Manager",
            "description": "Responsible for defining user personas, gathering customer requirements, and writing user stories. Handles queries about user needs, pain points, and desired outcomes from the end-user perspective.",
            "func": product_manager_support_function
        },
        {
            "name": "Program Manager",
            "description": "Responsible for defining features, epics, and technical capabilities. Handles queries about feature scope, system functionality, dependencies, and translating user needs into technical specifications.",
            "func": program_manager_support_function
        },
        {
            "name": "Development Engineer",
            "description": "Responsible for breaking down features into technical tasks, implementation details, and coding activities. Handles queries about development work, testing, debugging, and technical execution.",
            "func": development_engineer_support_function
        }
    ]
)

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = """Generate a project plan for the Email Router product with exactly 3 steps:
1. Write user stories for the Email Router product.
2. Define the product features for the Email Router product.
3. Define the engineering tasks for the Email Router product."""
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

# Cap at 3 steps in case the action planner over-splits the prompt
workflow_steps = [s for s in workflow_steps if s.strip()][:3]

completed_steps = []

for step in workflow_steps:
    print(f"\nExecuting step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"Result: {result[:200]}...")

print("\n*** Workflow execution completed ***\n")

section_labels = ["=== User Stories ===", "=== Product Features ===", "=== Engineering Tasks ==="]
final_output_parts = []
for i, step_result in enumerate(completed_steps):
    label = section_labels[i] if i < len(section_labels) else f"=== Step {i+1} ==="
    final_output_parts.append(f"{label}\n\n{step_result}")

final_output = "\n\n".join(final_output_parts)
print(final_output)

output_file = Path(__file__).parent / "workflow_results.txt"
with open(output_file, 'w') as f:
    f.write(final_output)
print(f"\nResults saved to: {output_file}")