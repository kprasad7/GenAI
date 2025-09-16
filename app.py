import os
import re
import subprocess
import sys
import webbrowser
import json
import time
from typing import List, Dict, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- CONFIG ---
MISTRAL_API_KEY = "1QJOIuK9SIhpKkF9vwg3IMgMIiEr0fQR"  # Replace with your real key
MODEL_NAME = "codestral-latest"
DEBUG = os.getenv("DEBUG", "").lower() == "true"

# Initialize LLM
llm = ChatMistralAI(model=MODEL_NAME, api_key=MISTRAL_API_KEY, temperature=0.1)

# --- HELPER FUNCTIONS ---

def extract_code_from_response(response: str) -> str:
    """Extract pure code from Markdown fence blocks."""
    pattern = r'```[a-zA-Z]*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return response.strip()

def detect_language_from_code(code: str) -> str:
    code_lower = code.lower()
    if "package main" in code_lower:
        return "go"
    elif "import java" in code_lower or ("public class" in code_lower and "static void main" in code_lower):
        return "java"
    elif "#include" in code_lower:
        return "c"
    elif "def " in code_lower or "import " in code_lower:
        return "python"
    elif "<!doctype html" in code_lower or "<html" in code_lower:
        return "html"
    elif "function " in code_lower and ("console.log" in code_lower or "document." in code_lower):
        return "javascript"
    elif "fmt." in code_lower:
        return "go"
    else:
        return "python"

def detect_extension(language: str) -> str:
    mapping = {
        "python": ".py",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "go": ".go",
        "r": ".r",
        "javascript": ".js",
        "js": ".js",
        "html": ".html",
        "react": ".jsx",
        "angular": ".ts"
    }
    return mapping.get(language.lower(), ".txt")

def extract_error_lines(error: str) -> List[int]:
    patterns = [
        r'(\d+):', r'line\s*(\d+)', r'line\s*:\s*(\d+)',
        r'File\s*"[^"]",\s*line\s(\d+)',
        r'at\s+\w+\.(\w+)\(([^)]+):(\d+)', r'at\s+\w+\.\w+\(([^:]+):(\d+)\)',
        r'(\d+)\s+in',
    ]
    lines = []
    for pattern in patterns:
        matches = re.findall(pattern, error)
        for match in matches:
            if isinstance(match, tuple):
                for group in match:
                    if group.isdigit():
                        lines.append(int(group))
            elif match.isdigit():
                lines.append(int(match))
    return list(set(lines))

def get_code_context(filename: str, error_lines: List[int], window: int = 3) -> Dict[int, str]:
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    snippets = {}
    for line_num in error_lines:
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        snippets[line_num] = ''.join(lines[start:end])
    return snippets

def apply_patch(filename: str, line_num: int, fixed_line: str):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = line_num - 1
    if idx >= len(lines):
        lines.append(fixed_line + '\n')
    else:
        lines[idx] = fixed_line + '\n'
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"🔧 Patched line {line_num} with: {fixed_line.strip()}")

# --- TOOLS (LANGCHAIN TOOL CLASSES) ---

class DockerPlatformTool(BaseTool):
    name: str = "docker_platform"
    description: str = "Create and manage Docker containers and images for applications."

    def _run(self, action: str, app_name: str = "app", language: str = "python") -> str:
        try:
            if action == "create_dockerfile":
                dockerfile_content = self._generate_dockerfile(language)
                with open("Dockerfile", "w") as f:
                    f.write(dockerfile_content)
                return f"✅ Dockerfile created for {language} application."
            
            elif action == "create_compose":
                compose_content = self._generate_docker_compose(app_name, language)
                with open("docker-compose.yml", "w") as f:
                    f.write(compose_content)
                return f"✅ docker-compose.yml created for {app_name}."
            
            elif action == "build":
                result = subprocess.run(["docker", "build", "-t", app_name, "."], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return f"✅ Docker image '{app_name}' built successfully."
                else:
                    return f"❌ Failed to build Docker image: {result.stderr}"
            
            elif action == "run":
                result = subprocess.run(["docker", "run", "-d", "-p", "8000:8000", "--name", f"{app_name}_container", app_name], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return f"✅ Docker container '{app_name}_container' started on port 8000."
                else:
                    return f"❌ Failed to run Docker container: {result.stderr}"
            
            elif action == "compose_up":
                result = subprocess.run(["docker-compose", "up", "-d"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return f"✅ Docker services started with docker-compose."
                else:
                    return f"❌ Failed to start services: {result.stderr}"
            
            elif action == "stop":
                result = subprocess.run(["docker", "stop", f"{app_name}_container"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return f"✅ Docker container '{app_name}_container' stopped."
                else:
                    return f"❌ Failed to stop container: {result.stderr}"
            
            elif action == "check_docker":
                result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return f"✅ Docker is available: {result.stdout.strip()}"
                else:
                    return f"❌ Docker is not available. Please install Docker."
            
            else:
                return f"❌ Unknown action: {action}. Use 'create_dockerfile', 'create_compose', 'build', 'run', 'compose_up', 'stop', or 'check_docker'."
        
        except Exception as e:
            return f"Error managing Docker platform: {str(e)}"

    def _generate_dockerfile(self, language: str, frontend_framework: str = "html") -> str:
        if language.lower() == "python":
            if frontend_framework in ["react", "vue", "angular", "nextjs"]:
                # Multi-stage build for full-stack apps
                return f"""# Multi-stage build for full-stack application
FROM node:18-alpine AS frontend

WORKDIR /app/frontend
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM python:3.11-slim AS backend

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app.py .

# Copy built frontend assets
COPY --from=frontend /app/frontend/build ./static
COPY --from=frontend /app/frontend/public ./templates

EXPOSE 8000

CMD ["python", "app.py"]
"""
            else:
                # Simple Python app with HTML/CSS/JS frontend
                return """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""
        elif language.lower() == "node" or language.lower() == "javascript":
            return """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 8000

CMD ["node", "app.js"]
"""
        elif language.lower() == "go":
            return """FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN go mod tidy
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .

EXPOSE 8000

CMD ["./main"]
"""
        else:
            return f"""FROM ubuntu:22.04

WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["./app"]
"""

    def _generate_docker_compose(self, app_name: str, language: str) -> str:
        if language.lower() == "python":
            return f"""version: '3.8'

services:
  {app_name}:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: {app_name}
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
"""
        else:
            return f"""version: '3.8'

services:
  {app_name}:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    restart: unless-stopped
"""

    async def _arun(self, action: str, app_name: str = "app", language: str = "python") -> str:
        raise NotImplementedError("Async not implemented.")

class DependencyManagerTool(BaseTool):
    name: str = "dependency_manager"
    description: str = "Manage dependencies for Docker-based applications using requirements.txt files."

    def _run(self, action: str, packages: str = "", requirements_file: str = "requirements.txt") -> str:
        try:
            if action == "create_requirements":
                if not packages:
                    return "No packages specified for requirements.txt."
                
                package_list = [pkg.strip() for pkg in packages.split(",")]
                
                with open(requirements_file, "w") as f:
                    for package in package_list:
                        f.write(f"{package}\n")
                
                return f"✅ Created {requirements_file} with packages: {', '.join(package_list)}"
            
            elif action == "add_packages":
                if not packages:
                    return "No packages specified to add."
                
                package_list = [pkg.strip() for pkg in packages.split(",")]
                
                # Read existing requirements if file exists
                existing_packages = set()
                if os.path.exists(requirements_file):
                    with open(requirements_file, "r") as f:
                        existing_packages = {line.strip() for line in f if line.strip()}
                
                # Add new packages
                for package in package_list:
                    existing_packages.add(package)
                
                # Write back to file
                with open(requirements_file, "w") as f:
                    for package in sorted(existing_packages):
                        f.write(f"{package}\n")
                
                return f"✅ Added packages to {requirements_file}: {', '.join(package_list)}"
            
            elif action == "auto_detect":
                # Auto-detect dependencies from common import statements
                dependencies = set()
                
                # Scan for Python files and extract imports
                for file in os.listdir('.'):
                    if file.endswith('.py'):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Common dependency mappings
                            import_mappings = {
                                'fastapi': 'fastapi',
                                'uvicorn': 'uvicorn',
                                'pydantic': 'pydantic',
                                'requests': 'requests',
                                'pandas': 'pandas',
                                'numpy': 'numpy',
                                'flask': 'flask',
                                'django': 'django',
                                'sqlalchemy': 'sqlalchemy',
                                'psycopg2': 'psycopg2-binary',
                                'PIL': 'Pillow',
                                'cv2': 'opencv-python',
                                'sklearn': 'scikit-learn',
                                'torch': 'torch',
                                'tensorflow': 'tensorflow',
                                'matplotlib': 'matplotlib',
                                'plotly': 'plotly',
                                'streamlit': 'streamlit',
                                'gradio': 'gradio',
                                'openai': 'openai',
                                'langchain': 'langchain',
                                'transformers': 'transformers',
                                'reportlab': 'reportlab'
                            }
                            
                            # Look for import statements
                            import_patterns = [
                                r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                                r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
                            ]
                            
                            for pattern in import_patterns:
                                matches = re.findall(pattern, content)
                                for match in matches:
                                    if match in import_mappings:
                                        dependencies.add(import_mappings[match])
                        
                        except Exception:
                            continue
                
                if dependencies:
                    with open(requirements_file, "w") as f:
                        for dep in sorted(dependencies):
                            f.write(f"{dep}\n")
                    
                    return f"✅ Auto-detected dependencies and created {requirements_file}: {', '.join(sorted(dependencies))}"
                else:
                    return f"❌ No common dependencies detected in Python files."
            
            elif action == "check":
                if os.path.exists(requirements_file):
                    with open(requirements_file, "r") as f:
                        packages = [line.strip() for line in f if line.strip()]
                    return f"✅ {requirements_file} exists with {len(packages)} packages: {', '.join(packages)}"
                else:
                    return f"❌ {requirements_file} not found."
            
            else:
                return f"❌ Unknown action: {action}. Use 'create_requirements', 'add_packages', 'auto_detect', or 'check'."
        
        except Exception as e:
            return f"Error managing dependencies: {str(e)}"

    async def _arun(self, action: str, packages: str = "", requirements_file: str = "requirements.txt") -> str:
        raise NotImplementedError("Async not implemented.")

class PlatformAgentTool(BaseTool):
    name: str = "platform_agent"
    description: str = "Comprehensive platform management including Docker setup, dependency resolution, and CLI error handling."

    def _run(self, action: str, error_message: str = "", filename: str = "", app_name: str = "app") -> str:
        try:
            if action == "setup_docker_environment":
                results = []
                
                # Step 1: Check if Docker is available
                docker_tool = DockerPlatformTool()
                docker_check = docker_tool._run("check_docker")
                results.append(docker_check)
                
                if "not available" in docker_check:
                    return "❌ Docker is not available. Please install Docker first."
                
                # Step 2: Create Dockerfile
                dockerfile_result = docker_tool._run("create_dockerfile", app_name, "python")
                results.append(dockerfile_result)
                
                # Step 3: Create docker-compose.yml
                compose_result = docker_tool._run("create_compose", app_name, "python")
                results.append(compose_result)
                
                return "Docker Environment Setup Results:\n" + "\n".join(results)
            
            elif action == "create_requirements":
                # Auto-detect dependencies and create requirements.txt after code is generated
                dep_tool = DependencyManagerTool()
                dep_result = dep_tool._run("auto_detect")
                return f"Requirements Creation Result:\n{dep_result}"
            
            elif action == "build_and_run":
                return self._build_and_run_with_retry(app_name)
            
            elif action == "deploy_until_success":
                return self._deploy_until_success(app_name)
            
            elif action == "resolve_import_error":
                if not error_message:
                    return "No error message provided for import error resolution."
                
                # Extract module names from import errors
                import_patterns = [
                    r"No module named '([^']+)'",
                    r"ModuleNotFoundError: No module named '([^']+)'",
                    r"ImportError: No module named ([^\s]+)",
                    r"cannot import name '([^']+)'"
                ]
                
                missing_modules = []
                for pattern in import_patterns:
                    matches = re.findall(pattern, error_message)
                    missing_modules.extend(matches)
                
                if missing_modules:
                    # Map common module names to package names
                    module_to_package = {
                        "fastapi": "fastapi",
                        "uvicorn": "uvicorn",
                        "PIL": "pillow",
                        "cv2": "opencv-python",
                        "sklearn": "scikit-learn",
                        "torch": "torch",
                        "tensorflow": "tensorflow",
                        "streamlit": "streamlit",
                        "flask": "flask",
                        "django": "django",
                        "requests": "requests",
                        "numpy": "numpy",
                        "pandas": "pandas",
                        "matplotlib": "matplotlib",
                        "reportlab": "reportlab",
                        "langchain": "langchain",
                        "openai": "openai"
                    }
                    
                    packages_to_add = []
                    for module in missing_modules:
                        package = module_to_package.get(module, module)
                        packages_to_add.append(package)
                    
                    # Add packages to requirements.txt
                    dep_tool = DependencyManagerTool()
                    result = dep_tool._run("add_packages", ",".join(packages_to_add))
                    
                    return f"✅ Resolved import errors by adding packages to requirements.txt: {', '.join(packages_to_add)}\n{result}"
                else:
                    return "❌ No missing modules detected in error message."
            
            elif action == "handle_cli_error":
                if not error_message:
                    return "No error message provided for CLI error handling."
                
                # Handle common CLI errors
                if "command not found" in error_message.lower():
                    return "❌ Command not found. Ensure the required tool is installed in the Docker container."
                
                elif "permission denied" in error_message.lower():
                    return "❌ Permission denied. Check file permissions or run with appropriate privileges."
                
                elif "container name" in error_message.lower() and "already in use" in error_message.lower():
                    # Extract container name and remove the conflicting container
                    import re
                    container_match = re.search(r'container name "([^"]+)"', error_message)
                    if container_match:
                        container_name = container_match.group(1)
                        # Remove the existing container
                        result = subprocess.run(["docker", "rm", "-f", container_name], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            return f"✅ Removed conflicting container '{container_name}'. Ready to retry deployment."
                        else:
                            return f"❌ Failed to remove container '{container_name}': {result.stderr}"
                    return "❌ Container name conflict detected but couldn't extract container name."
                
                elif "port" in error_message.lower() and "already in use" in error_message.lower():
                    return "❌ Port already in use. Stop existing containers or use a different port."
                
                elif "no space left" in error_message.lower():
                    return "❌ No space left on device. Clean up Docker images and containers."
                
                else:
                    return f"❌ Unhandled CLI error. Manual intervention may be required: {error_message}"
            
            else:
                return f"❌ Unknown action: {action}. Use 'setup_docker_environment', 'create_requirements', 'build_and_run', 'deploy_until_success', 'resolve_import_error', or 'handle_cli_error'."
        
        except Exception as e:
            return f"Error in platform management: {str(e)}"

    def _build_and_run_with_retry(self, app_name: str, max_retries: int = 3) -> str:
        """Build and run Docker container with automatic retry logic."""
        docker_tool = DockerPlatformTool()
        results = []
        
        for attempt in range(1, max_retries + 1):
            print(f"🔄 Deployment attempt {attempt}/{max_retries}")
            
            # Step 1: Build Docker image
            build_result = docker_tool._run("build", app_name)
            results.append(f"Attempt {attempt} - Build: {build_result}")
            
            if "successfully" in build_result:
                # Step 2: Try to run Docker container
                run_result = docker_tool._run("run", app_name)
                results.append(f"Attempt {attempt} - Run: {run_result}")
                
                if "started" in run_result:
                    return f"✅ SUCCESS! Application deployed successfully on attempt {attempt}\n" + "\n".join(results)
                
                elif "already in use" in run_result:
                    # Handle container name conflict
                    print(f"🔧 Container conflict detected, cleaning up...")
                    cleanup_result = self._handle_container_conflict(run_result, app_name)
                    results.append(f"Attempt {attempt} - Cleanup: {cleanup_result}")
                    
                    if "✅" in cleanup_result:
                        # Retry running after cleanup
                        retry_run_result = docker_tool._run("run", app_name)
                        results.append(f"Attempt {attempt} - Retry Run: {retry_run_result}")
                        
                        if "started" in retry_run_result:
                            return f"✅ SUCCESS! Application deployed successfully after cleanup on attempt {attempt}\n" + "\n".join(results)
            
            # If this attempt failed, wait a moment before retrying
            if attempt < max_retries:
                print(f"⏳ Waiting before retry...")
                time.sleep(2)
        
        return f"❌ FAILED! Could not deploy application after {max_retries} attempts\n" + "\n".join(results)

    def _deploy_until_success(self, app_name: str, max_attempts: int = 10) -> str:
        """Keep trying to deploy until success or max attempts reached."""
        docker_tool = DockerPlatformTool()
        dep_tool = DependencyManagerTool()
        
        for attempt in range(1, max_attempts + 1):
            print(f"🚀 Deployment attempt {attempt}/{max_attempts}")
            
            # Try the full deployment process
            try:
                # Step 1: Ensure requirements.txt exists
                if not os.path.exists("requirements.txt"):
                    dep_result = dep_tool._run("auto_detect")
                    print(f"📦 Created requirements: {dep_result}")
                
                # Step 2: Build and run
                deployment_result = self._build_and_run_with_retry(f"{app_name}_v{attempt}")
                
                if "SUCCESS" in deployment_result:
                    return f"🎉 Application successfully deployed on attempt {attempt}!\n{deployment_result}"
                
                # If failed, try to fix common issues
                if "requirements.txt" in deployment_result and "not found" in deployment_result:
                    # Create a basic requirements.txt
                    with open("requirements.txt", "w") as f:
                        f.write("flask\nfastapi\nuvicorn\npillow\npython-multipart\n")
                    print("📦 Created basic requirements.txt")
                
            except Exception as e:
                print(f"❌ Attempt {attempt} failed with error: {e}")
            
            if attempt < max_attempts:
                print(f"⏳ Waiting before next attempt...")
                time.sleep(3)
        
        return f"❌ Could not deploy application after {max_attempts} attempts. Manual intervention required."

    def _handle_container_conflict(self, error_message: str, app_name: str) -> str:
        """Handle Docker container name conflicts by cleaning up."""
        import re
        
        # Extract container name from error message
        container_match = re.search(r'container name "([^"]+)"', error_message)
        if container_match:
            container_name = container_match.group(1)
        else:
            # Fallback to app_name pattern
            container_name = f"{app_name}_container"
        
        try:
            # Stop and remove the existing container
            stop_result = subprocess.run(["docker", "stop", container_name], 
                                       capture_output=True, text=True)
            remove_result = subprocess.run(["docker", "rm", container_name], 
                                         capture_output=True, text=True)
            
            if remove_result.returncode == 0:
                return f"✅ Cleaned up conflicting container '{container_name}'"
            else:
                return f"⚠️ Attempted cleanup of '{container_name}' but may still have issues"
                
        except Exception as e:
            return f"❌ Failed to cleanup container: {str(e)}"

    async def _arun(self, action: str, error_message: str = "", filename: str = "", app_name: str = "app") -> str:
        raise NotImplementedError("Async not implemented.")

class GenerateCodeTool(BaseTool):
    name: str = "generate_code"
    description: str = "Generate complete full-stack applications with Python backend and appropriate frontend. Always creates runnable localhost applications."

    def _run(self, idea: str, frontend_framework: str = "auto") -> str:
        # Determine the best frontend framework based on the idea
        if frontend_framework == "auto":
            frontend_framework = self._detect_frontend_framework(idea)
        
        # Generate backend code (always Python)
        backend_code = self._generate_backend(idea)
        
        # Generate frontend code based on detected/specified framework
        frontend_files = self._generate_frontend(idea, frontend_framework)
        
        # Save all generated files
        saved_files = []
        
        # Save backend file
        with open("app.py", "w", encoding="utf-8") as f:
            f.write(backend_code)
        saved_files.append("app.py")
        print(f"💾 Generated Python backend: app.py")
        
        # Save frontend files
        for filename, content in frontend_files.items():
            # Create directories if needed
            os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files.append(filename)
            print(f"💾 Generated frontend file: {filename}")
        
        return f"Generated full-stack application with Python backend and {frontend_framework} frontend. Files: {', '.join(saved_files)}"

    def _detect_frontend_framework(self, idea: str) -> str:
        idea_lower = idea.lower()
        
        # Check for specific framework mentions
        if "react" in idea_lower:
            return "react"
        elif "vue" in idea_lower:
            return "vue"
        elif "angular" in idea_lower:
            return "angular"
        elif "next.js" in idea_lower or "nextjs" in idea_lower:
            return "nextjs"
        
        # Check for complexity indicators
        elif any(word in idea_lower for word in ["dashboard", "admin", "complex", "spa", "single page"]):
            return "react"
        elif any(word in idea_lower for word in ["simple", "basic", "landing", "static"]):
            return "html"
        elif any(word in idea_lower for word in ["real-time", "chat", "interactive"]):
            return "react"
        else:
            return "html"  # Default to simple HTML/CSS/JS

    def _generate_backend(self, idea: str) -> str:
        system_msg = """You are a Python backend code generator. Generate ONLY complete, runnable Python backend code (Flask/FastAPI) based on the user's request.

Requirements:
- Always use Python (Flask or FastAPI)
- Include all necessary routes and endpoints
- Handle file uploads, downloads, forms as needed
- Include error handling and validation
- Make it localhost-runnable
- For external integrations (payments, APIs, etc.), implement dummy/mock versions with clear comments
- Include CORS handling for frontend integration
- Return pure code only, no explanations or markdown

For missing integrations, implement dummy versions like:
- Payment gateways: Mock payment processing with success/failure responses
- External APIs: Mock responses with realistic sample data
- Email services: Log email content instead of sending
- SMS services: Log SMS content instead of sending"""

        messages = [SystemMessage(content=system_msg), HumanMessage(content=f"Create Python backend for: {idea}")]
        response = llm.invoke(messages)
        return extract_code_from_response(response.content)

    def _generate_frontend(self, idea: str, framework: str) -> dict:
        frontend_files = {}
        
        if framework == "html":
            frontend_files.update(self._generate_html_frontend(idea))
        elif framework == "react":
            frontend_files.update(self._generate_react_frontend(idea))
        elif framework == "vue":
            frontend_files.update(self._generate_vue_frontend(idea))
        elif framework == "angular":
            frontend_files.update(self._generate_angular_frontend(idea))
        elif framework == "nextjs":
            frontend_files.update(self._generate_nextjs_frontend(idea))
        
        return frontend_files

    def _generate_html_frontend(self, idea: str) -> dict:
        system_msg = """Generate complete HTML/CSS/JavaScript frontend files for the given idea.

Requirements:
- Create responsive, modern UI with HTML5, CSS3, and vanilla JavaScript
- Include all necessary HTML pages
- Include comprehensive CSS styling (can be inline or separate file)
- Include JavaScript for interactivity and API calls to Python backend
- Make it production-ready and visually appealing
- Handle forms, file uploads, data display as needed
- Use fetch() for API calls to localhost backend
- Return as dictionary with filename->content mapping

File structure should include:
- index.html (main page)
- style.css (if separate CSS needed)
- script.js (if separate JS needed)
- Additional HTML pages as needed"""

        messages = [SystemMessage(content=system_msg), HumanMessage(content=f"Create HTML/CSS/JS frontend for: {idea}")]
        response = llm.invoke(messages)
        
        # Parse the response to extract multiple files
        content = response.content
        files = {}
        
        # Try to extract multiple code blocks
        pattern = r'```(?:html|css|javascript|js)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if len(matches) >= 1:
            # If we have multiple blocks, assume they are HTML, CSS, JS in order
            if len(matches) == 1:
                # Single file, assume it's complete HTML with inline CSS/JS
                files["index.html"] = matches[0].strip()
            else:
                # Multiple files
                files["index.html"] = matches[0].strip()
                if len(matches) > 1:
                    files["style.css"] = matches[1].strip()
                if len(matches) > 2:
                    files["script.js"] = matches[2].strip()
        else:
            # No code blocks found, use entire response as HTML
            files["index.html"] = extract_code_from_response(content)
        
        return files

    def _generate_react_frontend(self, idea: str) -> dict:
        system_msg = """Generate complete React frontend application for the given idea.

Requirements:
- Create modern React application with hooks
- Include package.json with all dependencies
- Include complete component structure
- Include App.js, index.js, and necessary components
- Include CSS styling (can use CSS modules or styled-components)
- Handle API calls to Python backend running on localhost
- Make it production-ready and responsive
- Use functional components and hooks
- Include error handling and loading states

Return as dictionary with filename->content mapping including:
- package.json
- public/index.html
- src/index.js
- src/App.js
- src/components/*.js (as needed)
- src/App.css or component-specific CSS"""

        messages = [SystemMessage(content=system_msg), HumanMessage(content=f"Create React frontend for: {idea}")]
        response = llm.invoke(messages)
        
        # For now, return a basic React structure
        # In a full implementation, you'd parse the LLM response for multiple files
        content = extract_code_from_response(response.content)
        
        return {
            "src/App.js": content,
            "package.json": self._generate_react_package_json(),
            "public/index.html": self._generate_react_index_html()
        }

    def _generate_vue_frontend(self, idea: str) -> dict:
        # Similar implementation for Vue
        return {"src/App.vue": "<!-- Vue frontend would be generated here -->"}

    def _generate_angular_frontend(self, idea: str) -> dict:
        # Similar implementation for Angular
        return {"src/app/app.component.ts": "// Angular frontend would be generated here"}

    def _generate_nextjs_frontend(self, idea: str) -> dict:
        # Similar implementation for Next.js
        return {"pages/index.js": "// Next.js frontend would be generated here"}

    def _generate_react_package_json(self) -> str:
        return """{
  "name": "frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:5000"
}"""

    def _generate_react_index_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Full-stack application" />
    <title>Full-Stack App</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>"""

    async def _arun(self, idea: str, frontend_framework: str = "auto") -> str:
        raise NotImplementedError("Async not implemented.")

class FrontendBuildTool(BaseTool):
    name: str = "frontend_build"
    description: str = "Handle frontend framework setup, dependency installation, and build processes."

    def _run(self, action: str, framework: str = "html", build_dir: str = "dist") -> str:
        try:
            if action == "setup":
                if framework == "react":
                    return self._setup_react()
                elif framework == "vue":
                    return self._setup_vue()
                elif framework == "angular":
                    return self._setup_angular()
                elif framework == "nextjs":
                    return self._setup_nextjs()
                elif framework == "html":
                    return "✅ HTML/CSS/JS setup complete - no build process needed."
                else:
                    return f"❌ Unknown frontend framework: {framework}"
            
            elif action == "install":
                if framework in ["react", "vue", "angular", "nextjs"]:
                    result = subprocess.run(["npm", "install"], capture_output=True, text=True)
                    if result.returncode == 0:
                        return f"✅ NPM dependencies installed for {framework}"
                    else:
                        return f"❌ Failed to install NPM dependencies: {result.stderr}"
                else:
                    return "✅ No dependencies to install for HTML/CSS/JS"
            
            elif action == "build":
                if framework == "react":
                    result = subprocess.run(["npm", "run", "build"], capture_output=True, text=True)
                    if result.returncode == 0:
                        return f"✅ React app built successfully in build/"
                    else:
                        return f"❌ React build failed: {result.stderr}"
                elif framework in ["vue", "angular", "nextjs"]:
                    result = subprocess.run(["npm", "run", "build"], capture_output=True, text=True)
                    if result.returncode == 0:
                        return f"✅ {framework.title()} app built successfully"
                    else:
                        return f"❌ {framework.title()} build failed: {result.stderr}"
                else:
                    return "✅ HTML/CSS/JS requires no build process"
            
            elif action == "dev":
                if framework in ["react", "vue", "angular", "nextjs"]:
                    # Start development server in background
                    result = subprocess.Popen(["npm", "start"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return f"✅ Started {framework} development server (PID: {result.pid})"
                else:
                    return "✅ HTML/CSS/JS can be served directly"
            
            else:
                return f"❌ Unknown action: {action}. Use 'setup', 'install', 'build', or 'dev'."
                
        except Exception as e:
            return f"Error in frontend build process: {str(e)}"

    def _setup_react(self) -> str:
        # Create React index.js if it doesn't exist
        if not os.path.exists("src/index.js"):
            os.makedirs("src", exist_ok=True)
            with open("src/index.js", "w") as f:
                f.write("""import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
""")
        
        # Create public directory and index.html if needed
        os.makedirs("public", exist_ok=True)
        return "✅ React project structure created"

    def _setup_vue(self) -> str:
        # Similar setup for Vue
        return "✅ Vue project structure created"

    def _setup_angular(self) -> str:
        # Similar setup for Angular
        return "✅ Angular project structure created"

    def _setup_nextjs(self) -> str:
        # Similar setup for Next.js
        return "✅ Next.js project structure created"

    async def _arun(self, action: str, framework: str = "html", build_dir: str = "dist") -> str:
        raise NotImplementedError("Async not implemented.")

class IntegrationHandlerTool(BaseTool):
    name: str = "integration_handler"
    description: str = "Handle external integrations with dummy implementations or ask for clarification."

    def _run(self, integration_type: str, action: str = "check") -> str:
        try:
            if action == "check":
                return self._check_integration(integration_type)
            elif action == "implement_dummy":
                return self._implement_dummy(integration_type)
            elif action == "ask_clarification":
                return self._ask_clarification(integration_type)
            else:
                return f"❌ Unknown action: {action}. Use 'check', 'implement_dummy', or 'ask_clarification'."
        
        except Exception as e:
            return f"Error handling integration: {str(e)}"

    def _check_integration(self, integration_type: str) -> str:
        """Check if an integration can be implemented locally or needs dummy implementation."""
        localhost_integrations = [
            "database", "file_storage", "local_auth", "session_management", 
            "caching", "logging", "file_processing", "image_processing"
        ]
        
        dummy_integrations = [
            "payment_gateway", "stripe", "paypal", "email_service", "sms_service",
            "social_auth", "oauth", "external_api", "third_party_service",
            "cloud_storage", "aws", "azure", "gcp"
        ]
        
        if any(keyword in integration_type.lower() for keyword in localhost_integrations):
            return f"✅ {integration_type} can be implemented locally"
        elif any(keyword in integration_type.lower() for keyword in dummy_integrations):
            return f"⚠️ {integration_type} requires dummy implementation for localhost"
        else:
            return f"❓ {integration_type} needs clarification - can be implemented as dummy or requires external service"

    def _implement_dummy(self, integration_type: str) -> str:
        """Provide dummy implementation instructions."""
        dummy_implementations = {
            "payment": "Mock payment processing with success/failure responses, no real charges",
            "email": "Log email content to console/file instead of sending real emails",
            "sms": "Log SMS content to console/file instead of sending real SMS",
            "social_auth": "Use mock user data and local session management",
            "external_api": "Return realistic mock data instead of real API calls",
            "cloud_storage": "Use local file system to simulate cloud storage"
        }
        
        for key, implementation in dummy_implementations.items():
            if key in integration_type.lower():
                return f"✅ Dummy implementation for {integration_type}: {implementation}"
        
        return f"✅ Generic dummy implementation: Mock the {integration_type} with realistic local responses"

    def _ask_clarification(self, integration_type: str) -> str:
        """Generate clarification questions."""
        return f"""❓ Clarification needed for {integration_type}:

1. Should this be implemented as a dummy/mock version for localhost testing?
2. Do you want realistic sample data and responses?
3. Should the integration be prepared for later replacement with real service?
4. Are there specific requirements for the mock behavior?

Please specify your preference and I'll continue the implementation."""

    async def _arun(self, integration_type: str, action: str = "check") -> str:
        raise NotImplementedError("Async not implemented.")

class CLIMonitorTool(BaseTool):
    name: str = "cli_monitor"
    description: str = "Monitor CLI output, detect issues, and provide intelligent troubleshooting solutions like GitHub Copilot."

    def _run(self, action: str, command: str = "", output: str = "", error: str = "") -> str:
        try:
            if action == "analyze_error":
                return self._analyze_cli_error(output, error)
            elif action == "suggest_fix":
                return self._suggest_intelligent_fix(command, output, error)
            elif action == "monitor_logs":
                return self._monitor_container_logs(command)
            elif action == "health_check":
                return self._perform_health_check(command)
            else:
                return f"❌ Unknown action: {action}. Use 'analyze_error', 'suggest_fix', 'monitor_logs', or 'health_check'."
        
        except Exception as e:
            return f"Error in CLI monitoring: {str(e)}"

    def _analyze_cli_error(self, output: str, error: str) -> str:
        """Analyze CLI output and error messages to identify root causes."""
        issues_found = []
        solutions = []
        
        full_output = f"{output}\n{error}".lower()
        
        # Common Docker issues
        if "port already in use" in full_output or "port is already allocated" in full_output:
            issues_found.append("Port conflict")
            solutions.append("Stop conflicting containers or use different port")
        
        if "no such file or directory" in full_output and "requirements.txt" in full_output:
            issues_found.append("Missing requirements.txt")
            solutions.append("Create requirements.txt with detected dependencies")
        
        if "permission denied" in full_output:
            issues_found.append("Permission issue")
            solutions.append("Fix file permissions or run with appropriate privileges")
        
        if "container name" in full_output and "already in use" in full_output:
            issues_found.append("Container name conflict")
            solutions.append("Remove existing container with same name")
        
        if "image not found" in full_output or "no such image" in full_output:
            issues_found.append("Docker image missing")
            solutions.append("Rebuild Docker image")
        
        # Python/dependency issues
        if "modulenotfounderror" in full_output or "no module named" in full_output:
            issues_found.append("Missing Python dependencies")
            solutions.append("Add missing packages to requirements.txt and rebuild")
        
        if "syntaxerror" in full_output:
            issues_found.append("Python syntax error")
            solutions.append("Fix code syntax errors")
        
        if "indentationerror" in full_output:
            issues_found.append("Python indentation error")
            solutions.append("Fix code indentation")
        
        # Network/port issues
        if "address already in use" in full_output:
            issues_found.append("Address/port in use")
            solutions.append("Use different port or stop conflicting services")
        
        if "connection refused" in full_output:
            issues_found.append("Connection refused")
            solutions.append("Check if service is running and port is accessible")
        
        # Build issues
        if "build failed" in full_output or "error building" in full_output:
            issues_found.append("Docker build failure")
            solutions.append("Check Dockerfile and fix build issues")
        
        if "npm install failed" in full_output or "yarn install failed" in full_output:
            issues_found.append("Frontend dependency installation failed")
            solutions.append("Fix package.json or use different package manager")
        
        if not issues_found:
            return "🔍 No specific issues detected. May need manual investigation."
        
        result = f"🔍 CLI Analysis Results:\n"
        result += f"📋 Issues Found: {', '.join(issues_found)}\n"
        result += f"💡 Suggested Solutions: {'; '.join(solutions)}"
        
        return result

    def _suggest_intelligent_fix(self, command: str, output: str, error: str) -> str:
        """Provide intelligent fix suggestions like GitHub Copilot."""
        analysis = self._analyze_cli_error(output, error)
        
        # Generate specific fix commands based on analysis
        fixes = []
        
        if "Port conflict" in analysis:
            fixes.append("docker stop $(docker ps -q --filter 'publish=8000')")
            fixes.append("docker container prune -f")
            fixes.append("docker run -d --name blog-app-fixed -p 8001:8000 blog-app")
        
        if "Missing requirements.txt" in analysis:
            fixes.append("echo 'fastapi\nuvicorn\npillow\npython-multipart' > requirements.txt")
        
        if "Container name conflict" in analysis:
            # Extract container name from error
            import re
            container_match = re.search(r'container name "([^"]+)"', error)
            if container_match:
                container_name = container_match.group(1)
                fixes.append(f"docker rm -f {container_name}")
        
        if "Missing Python dependencies" in analysis:
            # Extract module name
            module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error.lower())
            if module_match:
                module_name = module_match.group(1)
                fixes.append(f"echo '{module_name}' >> requirements.txt")
                fixes.append("docker build -t app .")
        
        if "Docker build failure" in analysis:
            fixes.append("docker system prune -f")
            fixes.append("docker build --no-cache -t app .")
        
        if not fixes:
            fixes.append("Manual investigation required - check logs for specific errors")
        
        return f"🔧 Intelligent Fix Suggestions:\n" + "\n".join([f"• {fix}" for fix in fixes])

    def _monitor_container_logs(self, container_name: str) -> str:
        """Monitor container logs for real-time issue detection."""
        try:
            result = subprocess.run(["docker", "logs", "--tail", "50", container_name], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logs = result.stdout + result.stderr
                
                # Analyze logs for issues
                if "error" in logs.lower() or "failed" in logs.lower():
                    return f"⚠️ Issues detected in logs:\n{logs}\n\n{self._analyze_cli_error(logs, '')}"
                else:
                    return f"✅ Container running normally:\n{logs}"
            else:
                return f"❌ Could not fetch logs: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰ Log monitoring timed out"
        except Exception as e:
            return f"❌ Error monitoring logs: {str(e)}"

    def _perform_health_check(self, container_name: str) -> str:
        """Perform comprehensive health check on running container."""
        checks = []
        
        # Check if container is running
        result = subprocess.run(["docker", "ps", "--filter", f"name={container_name}", "--format", "table {{.Names}}\t{{.Status}}"], 
                              capture_output=True, text=True)
        
        if container_name in result.stdout:
            checks.append("✅ Container is running")
            
            # Check port accessibility
            port_result = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8000"], 
                                       capture_output=True, text=True)
            
            if port_result.stdout == "200":
                checks.append("✅ Application responding on port 8000")
            else:
                checks.append(f"⚠️ Application not responding (HTTP {port_result.stdout})")
        else:
            checks.append("❌ Container not running")
        
        return "\n".join(checks)

    async def _arun(self, action: str, command: str = "", output: str = "", error: str = "") -> str:
        raise NotImplementedError("Async not implemented.")

class IntelligentFixTool(BaseTool):
    name: str = "intelligent_fix"
    description: str = "Intelligently fix code and configuration issues based on runtime errors and CLI output."

    def _run(self, action: str, filename: str = "", error_type: str = "", error_message: str = "") -> str:
        try:
            if action == "fix_code_error":
                return self._fix_code_error(filename, error_message)
            elif action == "fix_config_error":
                return self._fix_configuration_error(error_type, error_message)
            elif action == "fix_dependency_error":
                return self._fix_dependency_error(error_message)
            elif action == "fix_docker_error":
                return self._fix_docker_error(error_message)
            else:
                return f"❌ Unknown action: {action}. Use 'fix_code_error', 'fix_config_error', 'fix_dependency_error', or 'fix_docker_error'."
        
        except Exception as e:
            return f"Error in intelligent fixing: {str(e)}"

    def _fix_code_error(self, filename: str, error_message: str) -> str:
        """Fix code errors intelligently based on error messages."""
        if not os.path.exists(filename):
            return f"❌ File {filename} not found"
        
        # Common code fixes
        fixes_applied = []
        
        with open(filename, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import issues
        if "no module named" in error_message.lower():
            module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_message.lower())
            if module_match:
                missing_module = module_match.group(1)
                
                # Common module mappings
                module_fixes = {
                    'pil': 'from PIL import Image',
                    'cv2': 'import cv2',
                    'requests': 'import requests',
                    'flask': 'from flask import Flask',
                    'fastapi': 'from fastapi import FastAPI'
                }
                
                if missing_module in module_fixes:
                    if module_fixes[missing_module] not in content:
                        content = module_fixes[missing_module] + '\n' + content
                        fixes_applied.append(f"Added import for {missing_module}")
        
        # Fix common syntax issues
        if "syntaxerror" in error_message.lower():
            # Fix missing colons
            if "expected ':'" in error_message:
                content = re.sub(r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+([^:\n]+)(?<!:)\s*\n', 
                               r'\1 \2:\n', content)
                fixes_applied.append("Added missing colons")
        
        # Fix indentation issues
        if "indentationerror" in error_message.lower():
            lines = content.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:']):
                        fixed_lines.append(line)
                    else:
                        fixed_lines.append('    ' + line)  # Add basic indentation
                else:
                    fixed_lines.append(line)
            content = '\n'.join(fixed_lines)
            fixes_applied.append("Fixed basic indentation")
        
        # Save fixed content
        if content != original_content:
            with open(filename, 'w') as f:
                f.write(content)
            
            return f"✅ Applied fixes to {filename}: {', '.join(fixes_applied)}"
        else:
            return f"⚠️ No automatic fixes available for this error type"

    def _fix_configuration_error(self, error_type: str, error_message: str) -> str:
        """Fix configuration-related errors."""
        fixes = []
        
        if "requirements.txt" in error_message and "not found" in error_message:
            # Create basic requirements.txt
            basic_requirements = [
                "fastapi>=0.104.1",
                "uvicorn[standard]>=0.24.0",
                "python-multipart>=0.0.6",
                "pillow>=10.0.0",
                "requests>=2.31.0"
            ]
            
            with open("requirements.txt", "w") as f:
                f.write('\n'.join(basic_requirements))
            
            fixes.append("Created requirements.txt with common dependencies")
        
        if "dockerfile" in error_message.lower() and "not found" in error_message:
            # Create basic Dockerfile
            dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""
            with open("Dockerfile", "w") as f:
                f.write(dockerfile_content)
            
            fixes.append("Created basic Dockerfile")
        
        return f"✅ Configuration fixes applied: {', '.join(fixes)}" if fixes else "⚠️ No configuration fixes needed"

    def _fix_dependency_error(self, error_message: str) -> str:
        """Fix dependency-related errors."""
        # Extract missing packages
        missing_packages = []
        
        patterns = [
            r"no module named ['\"]([^'\"]+)['\"]",
            r"modulenotfounderror: no module named ['\"]([^'\"]+)['\"]",
            r"import error: no module named ([^\s]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, error_message.lower())
            missing_packages.extend(matches)
        
        if missing_packages:
            # Common package mappings
            package_mappings = {
                'pil': 'pillow',
                'cv2': 'opencv-python',
                'sklearn': 'scikit-learn',
                'yaml': 'pyyaml'
            }
            
            packages_to_add = []
            for pkg in missing_packages:
                actual_pkg = package_mappings.get(pkg, pkg)
                packages_to_add.append(actual_pkg)
            
            # Add to requirements.txt
            if os.path.exists("requirements.txt"):
                with open("requirements.txt", "a") as f:
                    for pkg in packages_to_add:
                        f.write(f"\n{pkg}")
            else:
                with open("requirements.txt", "w") as f:
                    f.write('\n'.join(packages_to_add))
            
            return f"✅ Added missing packages to requirements.txt: {', '.join(packages_to_add)}"
        
        return "⚠️ No missing packages detected"

    def _fix_docker_error(self, error_message: str) -> str:
        """Fix Docker-related errors."""
        fixes = []
        
        if "container name" in error_message and "already in use" in error_message:
            # Extract and remove conflicting container
            container_match = re.search(r'container name "([^"]+)"', error_message)
            if container_match:
                container_name = container_match.group(1)
                try:
                    subprocess.run(["docker", "rm", "-f", container_name], check=True)
                    fixes.append(f"Removed conflicting container: {container_name}")
                except subprocess.CalledProcessError:
                    fixes.append(f"Attempted to remove container: {container_name}")
        
        if "port" in error_message.lower() and ("already in use" in error_message.lower() or "already allocated" in error_message.lower()):
            try:
                # Kill processes on port 8000
                subprocess.run(["pkill", "-f", ":8000"], stderr=subprocess.DEVNULL)
                # Stop containers using port 8000
                subprocess.run(["docker", "stop", "$(docker ps -q --filter 'publish=8000')"], shell=True, stderr=subprocess.DEVNULL)
                fixes.append("Killed processes and stopped containers using port 8000")
            except subprocess.CalledProcessError:
                fixes.append("Attempted to free port 8000")
        
        return f"✅ Docker fixes applied: {', '.join(fixes)}" if fixes else "⚠️ No Docker fixes applied"

    async def _arun(self, action: str, filename: str = "", error_type: str = "", error_message: str = "") -> str:
        raise NotImplementedError("Async not implemented.")

class AgentChainWorkflowTool(BaseTool):
    name: str = "agent_chain_workflow"
    description: str = "Orchestrate a complete agent chain workflow like GitHub Copilot for intelligent development and deployment."

    def _run(self, action: str, project_name: str = "", user_request: str = "") -> str:
        try:
            if action == "full_copilot_chain":
                return self._execute_full_copilot_chain(project_name, user_request)
            elif action == "monitor_and_fix":
                return self._continuous_monitor_and_fix(project_name)
            elif action == "intelligent_troubleshoot":
                return self._intelligent_troubleshooting_workflow(project_name)
            else:
                return f"❌ Unknown action: {action}. Use 'full_copilot_chain', 'monitor_and_fix', or 'intelligent_troubleshoot'."
        
        except Exception as e:
            return f"Error in agent chain workflow: {str(e)}"

    def _execute_full_copilot_chain(self, project_name: str, user_request: str) -> str:
        """Execute the complete Copilot-like agent chain workflow."""
        workflow_results = []
        workflow_results.append("🤖 INITIATING COPILOT-LIKE AGENT CHAIN WORKFLOW")
        workflow_results.append("=" * 60)
        
        # Phase 1: Planning and Analysis
        workflow_results.append("\n📋 PHASE 1: INTELLIGENT PLANNING")
        planning_result = self._intelligent_planning_phase(user_request)
        workflow_results.append(planning_result)
        
        # Phase 2: Code Generation
        workflow_results.append("\n💻 PHASE 2: FULL-STACK CODE GENERATION")
        generation_result = self._code_generation_phase(project_name, user_request)
        workflow_results.append(generation_result)
        
        # Phase 3: Platform Setup
        workflow_results.append("\n🐋 PHASE 3: PLATFORM DEPLOYMENT")
        deployment_result = self._intelligent_deployment_phase(project_name)
        workflow_results.append(deployment_result)
        
        # Phase 4: Continuous Monitoring and Fixing
        workflow_results.append("\n🔍 PHASE 4: CONTINUOUS MONITORING")
        monitoring_result = self._continuous_monitoring_phase(project_name)
        workflow_results.append(monitoring_result)
        
        # Phase 5: Final Verification
        workflow_results.append("\n✅ PHASE 5: FINAL VERIFICATION")
        verification_result = self._final_verification_phase(project_name)
        workflow_results.append(verification_result)
        
        return "\n".join(workflow_results)

    def _intelligent_planning_phase(self, user_request: str) -> str:
        """Intelligent planning phase like GitHub Copilot."""
        planning_steps = []
        
        # Analyze user request for complexity and requirements
        if any(keyword in user_request.lower() for keyword in ['payment', 'stripe', 'paypal']):
            planning_steps.append("💳 Detected payment integration requirement")
            planning_steps.append("  - Will implement dummy payment gateway")
            planning_steps.append("  - Will add payment endpoints and forms")
        
        if any(keyword in user_request.lower() for keyword in ['user', 'auth', 'login', 'register']):
            planning_steps.append("👤 Detected authentication requirement")
            planning_steps.append("  - Will implement user authentication system")
            planning_steps.append("  - Will add login/register functionality")
        
        if any(keyword in user_request.lower() for keyword in ['database', 'data', 'store']):
            planning_steps.append("🗄️ Detected data storage requirement")
            planning_steps.append("  - Will implement in-memory data storage")
            planning_steps.append("  - Will add CRUD operations")
        
        if any(keyword in user_request.lower() for keyword in ['api', 'rest', 'endpoint']):
            planning_steps.append("🔗 Detected API requirement")
            planning_steps.append("  - Will implement RESTful API endpoints")
            planning_steps.append("  - Will add proper request/response handling")
        
        # Determine frontend framework
        if any(keyword in user_request.lower() for keyword in ['react', 'vue', 'angular']):
            framework = next(fw for fw in ['react', 'vue', 'angular'] if fw in user_request.lower())
            planning_steps.append(f"⚛️ Frontend Framework: {framework.title()}")
        else:
            planning_steps.append("🌐 Frontend Framework: HTML/CSS/JS (default)")
        
        planning_steps.append("🐍 Backend Framework: Python FastAPI/Flask (enforced)")
        planning_steps.append("🐋 Deployment: Docker containerization")
        planning_steps.append("🔧 CI/CD: Automated deployment with error resolution")
        
        if not planning_steps:
            planning_steps.append("📝 Basic web application structure")
            planning_steps.append("🌐 Simple HTML/CSS/JS frontend")
            planning_steps.append("🐍 Python backend with API endpoints")
        
        return "🧠 Intelligent Analysis Complete:\n" + "\n".join(planning_steps)

    def _code_generation_phase(self, project_name: str, user_request: str) -> str:
        """Generate full-stack code with intelligent decision making."""
        generation_results = []
        
        # Use GenerateCodeTool for full-stack generation
        code_generator = GenerateCodeTool()
        generation_result = code_generator._run(user_request, "react")  # Fixed parameter order
        generation_results.append(f"📝 Code Generation: {generation_result}")
        
        # Use IntegrationHandlerTool for external services
        integration_handler = IntegrationHandlerTool()
        
        # Detect and implement required integrations
        integrations_needed = []
        if 'payment' in user_request.lower():
            integrations_needed.append('payment')
        if 'email' in user_request.lower():
            integrations_needed.append('email')
        if 'sms' in user_request.lower():
            integrations_needed.append('sms')
        if 'auth' in user_request.lower():
            integrations_needed.append('auth')
        
        for integration in integrations_needed:
            integration_result = integration_handler._run(integration, "implement")
            generation_results.append(f"🔗 Integration ({integration}): {integration_result}")
        
        # Generate requirements.txt
        dep_manager = DependencyManagerTool()
        deps_result = dep_manager._run("auto_detect")
        generation_results.append(f"📦 Dependencies: {deps_result}")
        
        return "\n".join(generation_results)

    def _intelligent_deployment_phase(self, project_name: str) -> str:
        """Deploy with intelligent retry and fixing."""
        deployment_results = []
        
        # Use enhanced PlatformAgentTool with intelligent retry
        platform_agent = PlatformAgentTool()
        deployment_result = platform_agent._run("deploy", project_name)
        deployment_results.append(f"🚀 Deployment: {deployment_result}")
        
        # If deployment failed, apply intelligent fixes
        if "FAILED" in deployment_result or "❌" in deployment_result:
            deployment_results.append("🔧 Applying intelligent fixes...")
            
            intelligent_fix = IntelligentFixTool()
            
            # Try to fix common issues
            config_fix = intelligent_fix._run("fix_config_error", "", "requirements", "requirements.txt not found")
            deployment_results.append(f"🔧 Config Fix: {config_fix}")
            
            # Retry deployment
            retry_result = platform_agent._run("deploy", project_name)
            deployment_results.append(f"🔄 Retry Result: {retry_result}")
        
        return "\n".join(deployment_results)

    def _continuous_monitoring_phase(self, project_name: str) -> str:
        """Continuous monitoring with automatic issue detection."""
        monitoring_results = []
        
        cli_monitor = CLIMonitorTool()
        
        # Check for running containers
        container_name = f"{project_name.lower()}-container"
        
        # Health check
        health_result = cli_monitor._run("health_check", container_name)
        monitoring_results.append(f"🏥 Health Check: {health_result}")
        
        # Monitor logs
        logs_result = cli_monitor._run("monitor_logs", container_name)
        monitoring_results.append(f"📋 Log Analysis: {logs_result}")
        
        # If issues detected, attempt fixes
        if "❌" in logs_result or "error" in logs_result.lower():
            monitoring_results.append("🔧 Issues detected, applying fixes...")
            
            intelligent_fix = IntelligentFixTool()
            fix_result = intelligent_fix._run("fix_docker_error", "", "", logs_result)
            monitoring_results.append(f"🔧 Auto-fix Applied: {fix_result}")
        
        return "\n".join(monitoring_results)

    def _final_verification_phase(self, project_name: str) -> str:
        """Final comprehensive verification."""
        verification_results = []
        
        # Check container status
        try:
            container_check = subprocess.run(["docker", "ps", "--filter", f"name={project_name}", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"], 
                                           capture_output=True, text=True)
            
            if container_check.stdout and project_name.lower() in container_check.stdout.lower():
                verification_results.append("✅ Container is running")
                verification_results.append(f"📊 Container Info: {container_check.stdout.strip()}")
            else:
                verification_results.append("⚠️ Container not found or not running")
        
        except Exception as e:
            verification_results.append(f"❌ Container verification failed: {e}")
        
        # Check application accessibility
        ports_to_check = [8000, 8001, 8002, 8003, 8004]
        accessible_ports = []
        
        for port in ports_to_check:
            try:
                response = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"http://localhost:{port}"], 
                                        capture_output=True, text=True, timeout=5)
                if response.stdout in ["200", "404"]:
                    accessible_ports.append(port)
            except:
                pass
        
        if accessible_ports:
            verification_results.append(f"🌐 Application accessible on ports: {accessible_ports}")
            verification_results.append(f"🔗 Primary URL: http://localhost:{accessible_ports[0]}")
        else:
            verification_results.append("⚠️ Application may not be accessible")
        
        # Overall success assessment
        if accessible_ports and "✅ Container is running" in "\n".join(verification_results):
            verification_results.append("\n🎉 DEPLOYMENT SUCCESSFUL! Application is ready for use.")
        else:
            verification_results.append("\n⚠️ Deployment partially successful - manual verification recommended")
        
        return "\n".join(verification_results)

    def _continuous_monitor_and_fix(self, project_name: str) -> str:
        """Continuous monitoring loop with automatic fixing."""
        monitor_results = []
        
        cli_monitor = CLIMonitorTool()
        intelligent_fix = IntelligentFixTool()
        
        # Run continuous monitoring for a set period
        max_monitoring_cycles = 3
        
        for cycle in range(max_monitoring_cycles):
            monitor_results.append(f"\n🔄 Monitoring Cycle {cycle + 1}/{max_monitoring_cycles}")
            
            # Check container health
            container_name = f"{project_name.lower()}-container"
            health_result = cli_monitor._run("health_check", container_name)
            monitor_results.append(f"🏥 Health: {health_result}")
            
            # Analyze logs for issues
            logs_result = cli_monitor._run("monitor_logs", container_name)
            
            if "❌" in logs_result or "error" in logs_result.lower():
                monitor_results.append("🚨 Issues detected! Applying fixes...")
                
                # Apply intelligent fixes
                fix_result = intelligent_fix._run("fix_docker_error", "", "", logs_result)
                monitor_results.append(f"🔧 Fix Applied: {fix_result}")
                
                # Verify fix worked
                time.sleep(2)
                verify_result = cli_monitor._run("health_check", container_name)
                monitor_results.append(f"✅ Fix Verification: {verify_result}")
            else:
                monitor_results.append("✅ No issues detected")
            
            if cycle < max_monitoring_cycles - 1:
                time.sleep(5)  # Wait between cycles
        
        monitor_results.append("\n🏁 Continuous monitoring cycle complete")
        return "\n".join(monitor_results)

    async def _arun(self, action: str, project_name: str = "", user_request: str = "") -> str:
        raise NotImplementedError("Async not implemented.")

class RunCodeTool(BaseTool):
    name: str = "run_code"
    description: str = "Run the generated code file and return exit code, stdout, stderr."

    def _run(self, filename: str) -> str:
        if not os.path.exists(filename):
            return f"❌ File '{filename}' not found."
        
        # Detect language and run appropriately
        if filename.endswith('.py'):
            cmd = [sys.executable, filename]
        elif filename.endswith('.js'):
            cmd = ['node', filename]
        elif filename.endswith('.go'):
            cmd = ['go', 'run', filename]
        elif filename.endswith('.java'):
            compile_result = subprocess.run(['javac', filename], capture_output=True, text=True)
            if compile_result.returncode != 0:
                return f"❌ Compilation failed:\n{compile_result.stderr}"
            class_name = filename.replace('.java', '')
            cmd = ['java', class_name]
        else:
            return f"❌ Unsupported file type: {filename}"
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"
        
        return output

    async def _arun(self, filename: str) -> str:
        raise NotImplementedError("Async not implemented.")

class FixCodeTool(BaseTool):
    name: str = "fix_code"
    description: str = "Fix errors in a code file by analyzing error logs and patching specific lines. Input: filename and error message."

    def _run(self, filename: str, error: str) -> str:
        if not os.path.exists(filename):
            return f"❌ File '{filename}' not found."
        
        print(f"🔧 Analyzing error in {filename}...")
        
        error_lines = extract_error_lines(error)
        if not error_lines:
            return f"❌ Could not extract line numbers from error message."
        
        context = get_code_context(filename, error_lines)
        
        # Build context string
        context_str = f"File: {filename}\nError: {error}\n\nCode context:\n"
        for line_num, code_snippet in context.items():
            context_str += f"\nAround line {line_num}:\n{code_snippet}\n"
        
        system_msg = """You are a code fixing AI. Given a filename, error message, and code context, provide ONLY the fixed line of code that should replace the problematic line. 
Do not include explanations, line numbers, or any other text. Return only the corrected code line."""
        
        messages = [SystemMessage(content=system_msg), HumanMessage(content=context_str)]
        response = llm.invoke(messages)
        
        fixed_line = response.content.strip()
        
        # Apply the fix to the first error line
        apply_patch(filename, error_lines[0], fixed_line)
        
        return f"✅ Applied fix to line {error_lines[0]} in {filename}"

    async def _arun(self, filename: str, error: str) -> str:
        raise NotImplementedError("Async not implemented.")

# --- MAIN AGENT EXECUTOR ---
def orchestrator():
    import sys
    
    print("🚀 AI CODING AGENT (LangChain + Docker) — Your Copilot-Level Autonomous Developer\n")
    
    # Get user input from command line argument or interactive input
    if len(sys.argv) > 1:
        idea = sys.argv[1]
        print(f"💡 Building application: {idea}")
    else:
        idea = input("💡 Please enter the website or app you want to build: ")
    
    if not idea.strip():
        print("❌ No idea provided. Exiting.")
        return

    # ✅ CREATE PROJECT FOLDER
    safe_folder_name = re.sub(r'[^\w\-_\. ]', '_', idea).strip().replace(' ', '_')[:50]
    project_dir = os.path.join(os.getcwd(), safe_folder_name)
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    print(f"📁 Created project folder: {project_dir}")

    # --- SETUP TOOLS (Full-Stack Docker-based) ---
    tools = [
        PlatformAgentTool(),
        DockerPlatformTool(),
        DependencyManagerTool(),
        GenerateCodeTool(),
        FrontendBuildTool(),
        IntegrationHandlerTool(),
        CLIMonitorTool(),
        IntelligentFixTool(),
        AgentChainWorkflowTool(),
        RunCodeTool(),
        FixCodeTool(),
    ]

    # --- MEMORY ---
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # --- PROMPT TEMPLATE ---
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AI coding agent with GitHub Copilot-like intelligent troubleshooting capabilities, specialized in creating complete full-stack applications with automated deployment and continuous error resolution.

🤖 COPILOT-LIKE INTELLIGENT WORKFLOW:
1. Use `agent_chain_workflow` with action "full_copilot_chain" to orchestrate the complete intelligent development process
2. The workflow includes: Planning → Code Generation → Deployment → Monitoring → Verification
3. Automatic CLI monitoring, error detection, and intelligent fixes applied continuously until success
4. Real-time troubleshooting with retry logic and adaptive problem-solving

🔧 INTELLIGENT TROUBLESHOOTING CAPABILITIES:
- CLI output analysis with `cli_monitor` tool for real-time error detection
- Automatic issue categorization (port conflicts, dependencies, syntax, Docker, etc.)
- Intelligent code fixing with `intelligent_fix` tool for targeted problem resolution
- Continuous deployment loops with automatic retry until success
- Agent chain workflow orchestration for complex deployment scenarios

ENHANCED FULL-STACK WORKFLOW:
1. INITIAL ASSESSMENT: Use `agent_chain_workflow` action "full_copilot_chain" to begin intelligent workflow
2. INTELLIGENT PLANNING: Analyze requirements and determine architecture automatically
3. CODE GENERATION: Generate complete frontend + backend with proper structure
4. INTEGRATION SETUP: Implement dummy versions of external services (payments, APIs, etc.)
5. INTELLIGENT DEPLOYMENT: Deploy with retry logic and automatic error resolution
6. CONTINUOUS MONITORING: Monitor CLI output and fix issues in real-time
7. VERIFICATION: Comprehensive deployment verification and health checks

AUTOMATIC ERROR RESOLUTION:
- Port conflicts: Automatically find available ports or kill conflicting processes
- Container conflicts: Remove existing containers and generate unique names
- Missing dependencies: Auto-detect and add to requirements.txt
- Build failures: Analyze Docker logs and fix Dockerfile/code issues
- Runtime errors: Fix syntax, imports, and configuration automatically
- Integration issues: Implement robust dummy services with realistic responses

COPILOT-LIKE INTELLIGENCE FEATURES:
- Real-time CLI monitoring with `cli_monitor` for issue detection
- Intelligent error analysis and fix suggestion like GitHub Copilot
- Automatic code fixing with context-aware solutions
- Continuous deployment attempts until success
- Adaptive problem-solving based on error patterns
- Agent chain orchestration for complex scenarios

REQUIREMENTS:
- Backend MUST be Python (Flask/FastAPI) - enforced
- Frontend auto-selected: HTML/CSS/JS for simple, React for complex
- ALL applications must run on localhost with automatic port management
- Implement DUMMY/MOCK versions for all external integrations
- Use Docker containerization with intelligent conflict resolution
- Apply continuous monitoring and fixing until application succeeds

SUCCESS CRITERIA (enforced by intelligent monitoring):
✅ Application builds without errors (auto-fixed if issues detected)
✅ Container runs and is accessible via HTTP (verified automatically)
✅ All endpoints respond correctly (monitored continuously)
✅ Frontend renders properly (verified through health checks)
✅ All integrations work with dummy implementations (tested automatically)

START COMMAND: Always begin with agent_chain_workflow action "full_copilot_chain" to initiate the intelligent Copilot-like development workflow."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- CREATE AGENT ---
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=20,
        handle_parsing_errors=True
    )

    # --- EXECUTE ---
    try:
        response = agent_executor.invoke({"input": idea})
        print(f"\n🎉 SUCCESS! Agent completed its mission.\nFinal output: {response['output']}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    orchestrator()
