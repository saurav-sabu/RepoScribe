from agno.models.groq import Groq
from agno.db.sqlite import SqliteDb
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools
from agno.tools.python import PythonTools
from agno.agent import Agent
from pathlib import Path
from dotenv import load_dotenv
import os
from agno.team import Team


# Load environment variables
load_dotenv()

# Model
model = Groq(id="qwen/qwen3-32b")

# Base directories
BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "repo"

# Create required directories
REPO_DIR.mkdir(parents=True, exist_ok=True)

# Database for agent memory
db = SqliteDb(
    db_file="memory.db",
    session_table="session_table"
)

# -----------------------------
# File Manager Agent
# -----------------------------
# file_manager_agent = Agent(
#     id="file_manager_agent",
#     name="File Manager Agent",
#     role="File system manager",
#     description="Manages files and directories inside the project workspace.",
#     model=model,
#     instructions=[
#         "You are an expert file management agent.",
#         "You can list directories and files inside the project workspace.",
#         "You can read file contents when explicitly asked.",
#         "You can write or update files only when instructed.",
#         "Operate strictly within the project directory."
#     ],
#     tools=[
#         FileTools(base_dir=REPO_DIR)
#     ]
# )

# -----------------------------
# GitHub Loader Agent
# -----------------------------
github_loader_agent = Agent(
    id="github_loader_agent",
    name="GitHub Loader Agent",
    role="Repository loader",
    description="Clones a GitHub repository into the local workspace.",
    instructions=[
        "You are responsible for cloning GitHub repositories.",
        "Use git clone to clone the repository provided as input.",
        "Clone the repository inside the 'repo' directory.",
        "Do not modify repository contents after cloning.",
        "Confirm successful cloning and list the top-level files."
    ],
    tools=[
        ShellTools(),
        FileTools(base_dir=REPO_DIR)
    ]
)

file_analyzer_agent = Agent(
    id="file_analyzer_agent",
    name="File Analyzer Agent",
    role="Repository analyzer",
    db=db,
    add_history_to_context=True,
    description=(
        "Analyzes the repository structure to identify project languages, "
        "frameworks, configuration files, and entry points for both "
        "JavaScript and Python projects."
    ),
    model=model,
    instructions=[
        "You are an expert repository analysis agent.",
        "Analyze the directory structure of the cloned repository.",
        
        # Language detection
        "Identify the primary programming languages used (JavaScript, TypeScript, Python, or others).",
        
        # JavaScript-specific detection
        "For JavaScript or TypeScript projects, detect frameworks and runtimes using indicators such as "
        "package.json, node_modules, src/, pages/, app/, and common entry files like index.js, app.js, or server.js.",
        
        # Python-specific detection
        "For Python projects, detect frameworks and runtimes using indicators such as "
        "requirements.txt, pyproject.toml, setup.py, Pipfile, and common entry files like main.py, app.py, or manage.py.",
        
        # Config & entry files
        "Locate and list important configuration files relevant to the detected languages.",
        "Identify application entry points based strictly on file presence and naming conventions.",
        
        # Safety & correctness
        "Do not assume application behavior beyond what is explicitly visible in the repository.",
        "If multiple languages or entry points are detected, clearly report them.",
        
        # Output & review
        "Present the analysis in a clear, structured, and human-readable summary.",
        "Explicitly ask the user to review and confirm or correct the analysis before any README generation proceeds."
    ],
    tools=[
        FileTools(base_dir=REPO_DIR)
    ]
)

code_agent = Agent(
    id="code_understanding_agent",
    name="Code Understanding Agent",
    role="Code comprehension specialist",
    db=db,
    add_history_to_context=True,
    description=(
        "Understands the purpose, core functionality, and exposed interfaces "
        "of the project by analyzing source code for JavaScript and Python repositories."
    ),
    model=model,
    instructions=[
        "You are an expert code comprehension agent.",
        "Analyze source code files to understand what the project does at a high level.",
        
        # Scope
        "Focus on publicly exposed functionality such as APIs, CLI commands, background jobs, or libraries.",
        "Identify core features strictly based on code structure, function names, routes, and imports.",
        
        # JavaScript-specific
        "For JavaScript or TypeScript code, identify API routes, middleware, CLI entry points, and environment variables using patterns like process.env.",
        
        # Python-specific
        "For Python code, identify API routes, CLI entry points, and environment variables using patterns like os.environ or settings files.",
        
        # Safety
        "Do not execute any code or infer runtime behavior.",
        "Do not assume business logic beyond what is explicitly present in the code.",
        
        # Output
        "Summarize findings in a clear, structured, and concise format suitable for README documentation.",
        "Explicitly ask the user to review and confirm the understanding before README generation continues."
    ],
    tools=[
        FileTools(base_dir=REPO_DIR)
    ]
)

readme_preview_agent = Agent(
    id="readme_preview_agent",
    name="README Preview Agent",
    role="Documentation writer",
    db=db,
    add_history_to_context=True,
    description=(
        "Generates a README.md preview for user review without writing it to disk."
    ),
    model=model,
    instructions=[
        "You are an expert technical documentation writer.",
        "Generate a complete README.md using GitHub-flavored Markdown.",
        "Use only information that has been reviewed and confirmed by the user.",
        "Do not invent or assume features, APIs, commands, or configurations.",
        
        "Include standard sections such as Project Overview, Features, Tech Stack, Installation, Usage, and Configuration.",
        "Adapt instructions based on the detected language (JavaScript or Python).",
        
        "Use clear headings, bullet points, fenced code blocks, and minimal emojis.",
        "DO NOT save the README.md to disk.",
        "Present the README content clearly and ask the user to approve or request changes."
    ]
)

readme_generation_team = Team(
    id="readme_generation_team",
    name="README Generation Team",
    role="Orchestration agent",
    description=(
        "An orchestration agent that coordinates multiple specialized agents "
        "to analyze a GitHub repository, answer repository-related questions, "
        "and generate an accurate README.md."
    ),
    model=model,
    db=db,
    add_history_to_context=True,
    instructions=[
        # Core orchestration role
        "You are the primary orchestration agent.",
        "You decide which agent to invoke based on the user's request.",
        "Do not blindly follow the README generation flow if the user asks a direct question.",

        # Repo loading
        "If the repository is not yet cloned and the user's request depends on repository contents, "
        "first invoke the GitHub Loader Agent and confirm successful cloning.",

        # Repo Q&A handling
        "If the user asks any question related to the repository, "
        "such as project purpose, architecture, languages used, entry points, APIs, configuration, "
        "setup steps, or usage, answer the question directly.",
        "Use the File Analyzer Agent or Code Understanding Agent as needed to gather accurate information.",
        "Base answers strictly on the repository contents and previously confirmed analysis.",
        "If the answer is unclear or not present in the repository, explicitly state that.",

        # README flow (only when relevant)
        "Proceed with README generation only when the user explicitly asks to generate or update the README.",
        "Follow the approved multi-step flow for README generation when applicable.",

        # Review & approval gates
        "Respect user confirmations and corrections at every step.",
        "Do not regenerate analysis or README unless the user requests it.",

        # Safety & correctness
        "Never hallucinate features, commands, or behavior.",
        "Do not assume intent; clarify if the user's request is ambiguous.",

        # Communication
        "Clearly state which agent is being used to answer the question when appropriate.",
        "Keep answers concise, technical, and easy to understand."
    ],
    members=[
        github_loader_agent,
        file_analyzer_agent,
        code_agent,
        readme_preview_agent,
    ]
)


if __name__ == "__main__":
    readme_generation_team.cli_app()

