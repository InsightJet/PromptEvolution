import asyncio
import json
import uuid
import threading
import httpx
from typing import Optional, Any
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import litellm
import gepa
from gepa import GEPAAdapter, EvaluationBatch
import base64

app = FastAPI(title="GEPA Prompt Evolution")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for evolution sessions
evolution_sessions = {}


class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key: str


class EvolutionConfig(BaseModel):
    seed_prompt: str  # Prompt X - the prompt to optimize
    judge_prompt: str  # Prompt Y - the evaluator prompt
    test_inputs: list[str]  # Sample inputs to test the prompt
    task_model: ModelConfig  # Model for running the prompt
    judge_model: ModelConfig  # Model for judging
    reflection_model: Optional[ModelConfig] = None
    max_iterations: int = 10
    population_size: int = 5


@dataclass
class TestInput:
    """Data instance for GEPA"""
    input_text: str
    id: int


@dataclass
class Trajectory:
    """Captures execution trace for reflection"""
    input_text: str
    output_text: str
    judge_feedback: str
    score: float


class EvolutionSession:
    def __init__(self, config: EvolutionConfig):
        self.id = str(uuid.uuid4())
        self.config = config
        self.current_iteration = 0
        self.status = "pending"
        self.candidates = []
        self.best_candidate = None
        self.best_score = 0
        self.initial_score = None  # Track seed prompt score
        self.logs = []
        self.is_running = False
        self.gepa_result = None
        self.template_mapping = {}  # Maps {{variableName}} -> actual substituted value

    def log(self, message: str, data: dict = None):
        entry = {"message": message, "data": data or {}}
        self.logs.append(entry)
        return entry


def get_litellm_model_string(config: ModelConfig) -> str:
    """Convert our model config to litellm format"""
    provider_map = {
        "openai": "",
        "anthropic": "anthropic/",
        "google": "gemini/",
        "mistral": "mistral/",
        "groq": "groq/",
    }
    prefix = provider_map.get(config.provider, f"{config.provider}/")
    return f"{prefix}{config.model}"


def extract_template_mapping(template: str, data: str) -> dict:
    """Extract mapping of template variables to their substituted values.

    Returns dict like: {'{{batchSize}}': '5', '{{componentDescriptions}}': '<div>...</div>'}
    """
    import re

    mapping = {}

    # Find all template variables
    double_brace_vars = re.findall(r'\{\{(\w+)\}\}', template)
    single_brace_vars = re.findall(r'\{(\w+)\}', template)
    single_brace_vars = [v for v in single_brace_vars if v not in ['id', 'type', 'intent', 'purpose', 'confidence']]

    # Calculate batchSize from data
    batch_size = max(1, len(re.findall(r'<[^>]+>', data)))
    if batch_size == 0:
        batch_size = len([l for l in data.split('\n') if l.strip()])

    for var in double_brace_vars:
        placeholder = '{{' + var + '}}'
        if var.lower() == 'batchsize':
            mapping[placeholder] = str(batch_size)
        else:
            mapping[placeholder] = data

    for var in single_brace_vars:
        if var not in ['id', 'type', 'intent', 'purpose', 'confidence']:
            placeholder = '{' + var + '}'
            if var.lower() == 'batchsize':
                mapping[placeholder] = str(batch_size)
            else:
                mapping[placeholder] = data

    return mapping


def restore_template_variables(evolved_prompt: str, template_mapping: dict) -> str:
    """Replace substituted values back with their template placeholders.

    Args:
        evolved_prompt: The evolved prompt with actual data
        template_mapping: Dict mapping placeholders to their substituted values

    Returns:
        Prompt with template variables restored
    """
    if not template_mapping:
        return evolved_prompt

    restored = evolved_prompt

    # Sort by value length (longest first) to avoid partial replacements
    sorted_mappings = sorted(template_mapping.items(), key=lambda x: len(x[1]), reverse=True)

    for placeholder, value in sorted_mappings:
        if value and len(value) > 1:  # Don't replace single characters
            # Try exact replacement first
            if value in restored:
                restored = restored.replace(value, placeholder, 1)  # Replace only first occurrence
                print(f"[Template Restore] Replaced value for {placeholder}")
            else:
                # Try trimmed version
                trimmed = value.strip()
                if trimmed and trimmed in restored:
                    restored = restored.replace(trimmed, placeholder, 1)
                    print(f"[Template Restore] Replaced trimmed value for {placeholder}")

    return restored


def fill_template(template: str, data: str) -> str:
    """Fill template variables with sample data.

    Supports {{variableName}} syntax. If no variables found,
    appends data to the template.
    """
    import re

    # Find all template variables - support both {{var}} and {var}
    double_brace_vars = re.findall(r'\{\{(\w+)\}\}', template)
    single_brace_vars = re.findall(r'\{(\w+)\}', template)

    # Filter out common false positives for single brace (like JSON examples)
    single_brace_vars = [v for v in single_brace_vars if v not in ['id', 'type', 'intent', 'purpose', 'confidence']]

    variables = double_brace_vars or single_brace_vars

    print(f"[DEBUG fill_template] Template preview: {template[:200]}...")
    print(f"[DEBUG fill_template] Double brace vars: {double_brace_vars}")
    print(f"[DEBUG fill_template] Single brace vars (filtered): {single_brace_vars}")
    print(f"[DEBUG fill_template] Data length: {len(data)}")

    if variables:
        filled = template

        # Calculate batchSize from data (count HTML-like elements or lines)
        batch_size = max(1, len(re.findall(r'<[^>]+>', data)))  # Count HTML tags
        if batch_size == 0:
            batch_size = len([l for l in data.split('\n') if l.strip()])  # Count non-empty lines

        for var in double_brace_vars:
            if var.lower() == 'batchsize':
                filled = filled.replace('{{' + var + '}}', str(batch_size))
            else:
                filled = filled.replace('{{' + var + '}}', data)

        for var in single_brace_vars:
            if var not in ['id', 'type', 'intent', 'purpose', 'confidence']:
                if var.lower() == 'batchsize':
                    filled = filled.replace('{' + var + '}', str(batch_size))
                else:
                    filled = filled.replace('{' + var + '}', data)

        print(f"[DEBUG fill_template] Batch size calculated: {batch_size}")
        print(f"[DEBUG fill_template] After filling: {filled[:300]}...")
        return filled
    else:
        # No template variables - append data to prompt
        print(f"[DEBUG fill_template] No variables found, appending data")
        return f"{template}\n\nInput:\n{data}"


def call_llm_sync(model_config: ModelConfig, prompt: str, user_input: str = None) -> str:
    """Synchronous LLM call for GEPA adapter.

    If user_input is provided, it fills template variables in prompt.
    """
    model_string = get_litellm_model_string(model_config)

    # Fill template if we have input data
    if user_input:
        final_prompt = fill_template(prompt, user_input)
    else:
        final_prompt = prompt

    response = litellm.completion(
        model=model_string,
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        api_key=model_config.api_key,
    )

    return response.choices[0].message.content


class LLMJudgeAdapter(GEPAAdapter[TestInput, Trajectory, str]):
    """
    Custom GEPA adapter that uses LLM-as-a-Judge for evaluation.

    - Prompt X (candidate["system_prompt"]) is the prompt being optimized
    - Prompt Y (judge_prompt) evaluates the outputs
    - Z (judge feedback) drives the evolution
    """

    def __init__(
        self,
        task_model: ModelConfig,
        judge_model: ModelConfig,
        judge_prompt: str,
        session: EvolutionSession
    ):
        self.task_model = task_model
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.session = session
        self.eval_count = 0

    def evaluate(
        self,
        batch: list[TestInput],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, str]:
        """
        Run Prompt X on inputs, then use Prompt Y (judge) to score.
        """
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        system_prompt = candidate.get("system_prompt", "")

        # Track evaluation count (not same as GEPA iterations)
        self.eval_count += 1
        # Don't override current_iteration here - let GEPAProgressLogger handle it

        self.session.log(f"Evaluation round {self.eval_count}: testing candidate on {len(batch)} inputs", {
            "prompt_preview": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
        })

        for item in batch:
            # Check if stopped
            if not self.session.is_running:
                self.session.log("Evolution stopped by user")
                raise StopIteration("Evolution stopped by user")

            try:
                # Fill template with sample data and run
                filled_prompt = fill_template(system_prompt, item.input_text)

                self.session.log(f"Running filled prompt", {
                    "sample_data_preview": item.input_text[:100] + "..." if len(item.input_text) > 100 else item.input_text,
                    "filled_prompt_preview": filled_prompt[:200] + "..." if len(filled_prompt) > 200 else filled_prompt
                })

                output = call_llm_sync(self.task_model, filled_prompt)
                outputs.append(output)

                # Create judge input - show the complete filled prompt
                judge_input = f"""
Prompt (with data filled in):
{filled_prompt}

Model Output:
{output}

Please evaluate the output quality and provide:
1. A score from 0-100
2. Specific feedback on what could be improved

Format your response as:
SCORE: [number]
FEEDBACK: [your detailed feedback]
"""

                # Run Prompt Y (judge) to get evaluation Z
                judge_response = call_llm_sync(self.judge_model, self.judge_prompt, judge_input)

                # Parse score
                score = 0.5  # default normalized score
                try:
                    for line in judge_response.split('\n'):
                        if line.strip().upper().startswith('SCORE:'):
                            score_str = line.split(':')[1].strip()
                            raw_score = float(score_str.replace('%', ''))
                            score = raw_score / 100.0  # Normalize to 0-1
                            break
                except:
                    pass

                scores.append(score)

                self.session.log(f"Test evaluation", {
                    "input": item.input_text[:50] + "...",
                    "output": output[:100] + "...",
                    "score": round(score * 100, 1)
                })

                if capture_traces:
                    trajectories.append(Trajectory(
                        input_text=item.input_text,
                        output_text=output,
                        judge_feedback=judge_response,
                        score=score
                    ))

            except Exception as e:
                self.session.log(f"Error evaluating input: {str(e)}")
                outputs.append("")
                scores.append(0.0)
                if capture_traces:
                    trajectories.append(Trajectory(
                        input_text=item.input_text,
                        output_text="",
                        judge_feedback=f"Error: {str(e)}",
                        score=0.0
                    ))

        avg_score = sum(scores) / len(scores) if scores else 0
        self.session.log(f"Batch evaluation complete", {"avg_score": round(avg_score * 100, 1)})

        # Track initial score (seed prompt's first evaluation)
        if self.session.initial_score is None:
            self.session.initial_score = avg_score * 100
            self.session.log("Initial seed prompt score", {"score": round(avg_score * 100, 1)})

        # Update session best score
        if avg_score > self.session.best_score:
            self.session.best_score = avg_score * 100
            self.session.best_candidate = system_prompt
            improvement = self.session.best_score - self.session.initial_score if self.session.initial_score else 0
            self.session.log("New best candidate!", {
                "score": round(avg_score * 100, 1),
                "improvement": round(improvement, 1)
            })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        evaluation_batch: EvaluationBatch[Trajectory, str],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Extract feedback from judge (Z) to guide prompt evolution.
        GEPA uses this to reflect on performance and propose improvements.
        """
        reflective_data = {}

        for component in components_to_update:
            examples = []
            if evaluation_batch.trajectories:
                for traj in evaluation_batch.trajectories:
                    examples.append({
                        "input": traj.input_text,
                        "output": traj.output_text,
                        "feedback": traj.judge_feedback,
                        "score": traj.score
                    })
            reflective_data[component] = examples

        self.session.log("Built reflective dataset for mutation", {
            "components": components_to_update,
            "num_examples": len(examples) if examples else 0
        })

        return reflective_data


class GEPAProgressLogger:
    """Custom logger to track GEPA progress and update session"""

    def __init__(self, session: EvolutionSession, max_iterations: int):
        self.session = session
        self.max_iterations = max_iterations
        self.iteration_count = 0

    def log(self, message: str):
        # Parse GEPA's log messages to track progress
        self.session.log(f"[GEPA] {message}")

        # Track iterations from GEPA messages - look for iteration patterns
        import re
        iter_match = re.search(r'[Ii]teration\s*[:#]?\s*(\d+)', message)
        if iter_match:
            iter_num = int(iter_match.group(1))
            self.session.current_iteration = min(iter_num, self.max_iterations)
        elif "iteration" in message.lower() and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            self.session.current_iteration = self.iteration_count

        # Track new candidates
        if "new candidate" in message.lower() or "proposed" in message.lower():
            self.session.log("New candidate proposed by GEPA")


def _populate_stopped_results(session: EvolutionSession, seed_prompt: str):
    """Populate session.candidates with available results when stopped early."""
    candidates = []

    # Add seed prompt with initial score if available
    if session.initial_score is not None:
        candidates.append({
            "prompt": seed_prompt,
            "score": session.initial_score,
            "index": 0,
            "label": "seed"
        })

    # Add best candidate if it's different from seed and has a score
    if session.best_candidate and session.best_score > 0:
        # Only add if different from seed
        if session.best_candidate != seed_prompt:
            candidates.append({
                "prompt": session.best_candidate,
                "score": session.best_score,
                "index": 1,
                "label": "best"
            })
        elif not candidates:
            # If best is same as seed but we have no other candidates, add it
            candidates.append({
                "prompt": session.best_candidate,
                "score": session.best_score,
                "index": 0
            })

    # Sort by score descending
    candidates.sort(key=lambda x: -x["score"])
    session.candidates = candidates

    if candidates:
        session.log("Returning results from stopped evolution", {
            "num_candidates": len(candidates),
            "best_score": session.best_score,
            "initial_score": session.initial_score,
            "improvement": round((session.best_score or 0) - (session.initial_score or 0), 1)
        })


def run_gepa_evolution(session: EvolutionSession):
    """Run GEPA optimization in a separate thread"""
    config = session.config
    session.status = "running"
    session.is_running = True

    try:
        # Prepare training data
        trainset = [
            TestInput(input_text=text, id=i)
            for i, text in enumerate(config.test_inputs)
        ]

        # Extract and store template mapping for later restoration
        # Use first test input to determine what values will be substituted
        if config.test_inputs:
            session.template_mapping = extract_template_mapping(
                config.seed_prompt,
                config.test_inputs[0]
            )
            session.log("Template mapping extracted", {
                "variables": list(session.template_mapping.keys())
            })

        # Create seed candidate
        seed_candidate = {
            "system_prompt": config.seed_prompt
        }

        # Create custom adapter with LLM-as-Judge
        adapter = LLMJudgeAdapter(
            task_model=config.task_model,
            judge_model=config.judge_model,
            judge_prompt=config.judge_prompt,
            session=session
        )

        # Determine reflection model
        reflection_model_config = config.reflection_model or config.judge_model
        reflection_lm = get_litellm_model_string(reflection_model_config)

        # Set API key for reflection model
        import os
        provider = reflection_model_config.provider
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = reflection_model_config.api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = reflection_model_config.api_key
        elif provider == "google":
            os.environ["GOOGLE_API_KEY"] = reflection_model_config.api_key
        elif provider == "groq":
            os.environ["GROQ_API_KEY"] = reflection_model_config.api_key

        # Create progress logger
        progress_logger = GEPAProgressLogger(session, config.max_iterations)

        session.log("Starting GEPA optimization", {
            "seed_prompt": config.seed_prompt[:100] + "...",
            "test_inputs": len(config.test_inputs),
            "max_iterations": config.max_iterations
        })

        # Calculate metric calls budget
        # Each iteration: evaluate candidates on all test inputs
        metric_budget = config.max_iterations * len(trainset) * 2  # *2 for mutations

        session.log(f"Metric budget: {metric_budget} calls")

        # Run GEPA optimization
        try:
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=trainset,  # Use same set for validation
                adapter=adapter,
                reflection_lm=reflection_lm,
                max_metric_calls=metric_budget,
                skip_perfect_score=False,  # Don't stop early on high scores
                perfect_score=1.0,  # Only stop if literally perfect
                display_progress_bar=False,
                logger=progress_logger,
                seed=42
            )
            session.gepa_result = result
        except StopIteration:
            # User stopped - use what we have
            session.log("Evolution stopped early by user")
            _populate_stopped_results(session, config.seed_prompt)
            session.status = "stopped"
            return
        except Exception as e:
            if "stopped" in str(e).lower():
                session.log("Evolution stopped early by user")
                _populate_stopped_results(session, config.seed_prompt)
                session.status = "stopped"
                return
            raise

        session.gepa_result = result

        # Store all candidates with scores, sorted by score descending
        all_candidates = [
            {"prompt": c.get("system_prompt", ""), "score": s * 100, "index": i}
            for i, (c, s) in enumerate(zip(result.candidates, result.val_aggregate_scores))
        ]
        all_candidates.sort(key=lambda x: (-x["score"], -x["index"]))  # Higher score first, later index wins ties
        session.candidates = all_candidates

        # Pick the LAST candidate with the best score (most evolved)
        # When scores tie, prefer evolved versions over seed
        if all_candidates:
            best = all_candidates[0]  # Already sorted: highest score, highest index for ties
            session.best_candidate = best["prompt"]
            session.best_score = best["score"]
        else:
            session.best_candidate = result.best_candidate.get("system_prompt", "")
            best_score = result.val_aggregate_scores[result.best_idx] if result.val_aggregate_scores else 0
            session.best_score = best_score * 100

        session.log("GEPA optimization completed!", {
            "best_score": session.best_score,
            "total_candidates": len(result.candidates),
            "all_scores": [c["score"] for c in all_candidates]
        })

        session.status = "completed"

    except Exception as e:
        session.status = "error"
        session.log(f"Error during GEPA optimization: {str(e)}")
        import traceback
        session.log(f"Traceback: {traceback.format_exc()}")


@app.post("/api/evolution/start")
async def start_evolution(config: EvolutionConfig):
    """Start a new GEPA evolution session"""
    session = EvolutionSession(config)
    evolution_sessions[session.id] = session

    # Run GEPA in background thread (it's synchronous)
    thread = threading.Thread(target=run_gepa_evolution, args=(session,))
    thread.start()

    return {"session_id": session.id}


@app.get("/api/evolution/{session_id}/status")
async def get_evolution_status(session_id: str):
    """Get current status of an evolution session"""
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_id]
    improvement = (session.best_score - session.initial_score) if session.initial_score else 0

    # Restore template variables in best_candidate
    best_candidate = session.best_candidate
    restored_candidate = None
    if best_candidate and session.template_mapping:
        restored_candidate = restore_template_variables(best_candidate, session.template_mapping)

    return {
        "id": session.id,
        "status": session.status,
        "current_iteration": session.current_iteration,
        "max_iterations": session.config.max_iterations,
        "best_score": session.best_score,
        "initial_score": session.initial_score,
        "improvement": round(improvement, 1),
        "best_candidate": best_candidate,
        "restored_candidate": restored_candidate,  # With {{variables}} restored
        "template_mapping": session.template_mapping,
        "candidates": session.candidates,
    }


@app.get("/api/evolution/{session_id}/stream")
async def stream_evolution(session_id: str):
    """Stream evolution progress via SSE"""
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_id]
    last_log_index = 0

    async def event_generator():
        nonlocal last_log_index

        while True:
            # Send new log entries
            while last_log_index < len(session.logs):
                log_entry = session.logs[last_log_index]
                yield {
                    "event": "log",
                    "data": json.dumps(log_entry)
                }
                last_log_index += 1

            # Send status update
            improvement = (session.best_score - session.initial_score) if session.initial_score else 0

            # Restore template variables
            best_candidate = session.best_candidate
            restored_candidate = None
            if best_candidate and session.template_mapping:
                restored_candidate = restore_template_variables(best_candidate, session.template_mapping)

            yield {
                "event": "status",
                "data": json.dumps({
                    "status": session.status,
                    "current_iteration": session.current_iteration,
                    "best_score": session.best_score,
                    "initial_score": session.initial_score,
                    "improvement": round(improvement, 1),
                    "best_candidate": best_candidate,
                    "restored_candidate": restored_candidate,
                    "template_mapping": session.template_mapping,
                    "candidates": session.candidates,
                })
            }

            if session.status in ["completed", "error", "stopped"]:
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post("/api/evolution/{session_id}/stop")
async def stop_evolution(session_id: str):
    """Stop an evolution session"""
    if session_id not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_id]
    session.is_running = False
    session.status = "stopped"
    return {"message": "Stop requested"}


@app.get("/api/providers")
async def get_providers():
    """Get list of supported model providers"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
            },
            {
                "id": "google",
                "name": "Google",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
            },
            {
                "id": "mistral",
                "name": "Mistral",
                "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
            },
            {
                "id": "groq",
                "name": "Groq",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "gemma2-9b-it"]
            }
        ]
    }


# ===== GENERATE TEST INPUTS =====

class GenerateTestInputsRequest(BaseModel):
    seed_prompt: str
    num_inputs: int = 5
    model: ModelConfig


@app.post("/api/generate-test-inputs")
async def generate_test_inputs(request: GenerateTestInputsRequest):
    """Generate relevant sample data based on the prompt template"""

    # Detect template variables like {{variableName}}
    import re
    variables = re.findall(r'\{\{(\w+)\}\}', request.seed_prompt)
    variables_hint = ""
    if variables:
        variables_hint = f"\n\nDetected template variables: {', '.join(variables)}. Generate sample data that would fill these variables."

    meta_prompt = f"""Analyze this prompt template and generate {request.num_inputs} diverse, realistic SAMPLE DATA sets that would be used with it.

The prompt template:
{request.seed_prompt}
{variables_hint}

Generate sample data that:
1. Matches what this template expects (e.g., if it classifies UI components, generate sample HTML/component data)
2. Covers different scenarios (simple, medium, complex cases)
3. Is realistic data that would actually be processed by this prompt
4. Each sample should be a complete, usable input

Return ONLY a JSON array of strings. Each string is one complete sample data set.
Example for a UI classifier: ["<button class='btn'>Submit</button>\\n<input type='text' placeholder='Email'>", "<div class='dropdown'>Select Option</div>"]"""

    try:
        model_string = get_litellm_model_string(request.model)

        response = litellm.completion(
            model=model_string,
            messages=[{"role": "user", "content": meta_prompt}],
            api_key=request.model.api_key,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON array from response
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            test_inputs = json.loads(match.group())
            return {"test_inputs": test_inputs}
        else:
            return {"test_inputs": [content]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate test inputs: {str(e)}")


# ===== GENERATE JUDGE =====

class GenerateJudgeRequest(BaseModel):
    seed_prompt: str  # The prompt to create a judge for
    additional_instructions: str = ""  # User's additional criteria
    model: ModelConfig  # Model to use for generation


@app.post("/api/generate-judge")
async def generate_judge_prompt(request: GenerateJudgeRequest):
    """Generate a judge/evaluator prompt based on the seed prompt and instructions"""

    meta_prompt = """You are an expert at creating evaluation criteria for AI prompts.

Given a prompt that will be optimized, create a detailed judge/evaluator prompt that can score outputs from that prompt.

The judge prompt should:
1. Be specific to what the original prompt is trying to accomplish
2. Have clear, measurable scoring criteria (0-100 total)
3. Include 4-5 distinct evaluation dimensions
4. Be strict but fair - most outputs should score 50-70, only exceptional ones above 85
5. Output format must be: SCORE: [number] followed by FEEDBACK: [detailed feedback]

Original Prompt to Evaluate:
{seed_prompt}

{additional_section}

Create a comprehensive judge prompt that evaluates outputs from the above prompt. The judge should be critical and differentiate between mediocre and excellent responses."""

    additional_section = ""
    if request.additional_instructions.strip():
        additional_section = f"""Additional Evaluation Requirements from User:
{request.additional_instructions}

Make sure to incorporate these specific requirements into the evaluation criteria."""

    full_meta_prompt = meta_prompt.format(
        seed_prompt=request.seed_prompt,
        additional_section=additional_section
    )

    try:
        model_string = get_litellm_model_string(request.model)

        response = litellm.completion(
            model=model_string,
            messages=[
                {"role": "user", "content": full_meta_prompt}
            ],
            api_key=request.model.api_key,
        )

        judge_prompt = response.choices[0].message.content
        return {"judge_prompt": judge_prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate judge: {str(e)}")


# ===== LANGFUSE INTEGRATION =====

class LangfuseConfig(BaseModel):
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"


class SavePromptRequest(BaseModel):
    langfuse: LangfuseConfig
    name: str
    prompt: str
    labels: list[str] = ["optimized"]
    commit_message: str = ""


def get_langfuse_auth_header(config: LangfuseConfig) -> dict:
    """Create Basic auth header for Langfuse API"""
    credentials = f"{config.public_key}:{config.secret_key}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@app.post("/api/langfuse/prompts")
async def list_langfuse_prompts(config: LangfuseConfig):
    """List all prompts from Langfuse"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.host}/api/public/v2/prompts",
                headers=get_langfuse_auth_header(config),
                params={"limit": 100}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Langfuse API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prompts: {str(e)}")


@app.post("/api/langfuse/prompts/{prompt_name}")
async def get_langfuse_prompt(prompt_name: str, config: LangfuseConfig, version: Optional[int] = None, label: Optional[str] = None):
    """Get a specific prompt from Langfuse"""
    try:
        params = {}
        if version:
            params["version"] = version
        if label:
            params["label"] = label

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.host}/api/public/v2/prompts/{prompt_name}",
                headers=get_langfuse_auth_header(config),
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Langfuse API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prompt: {str(e)}")


@app.post("/api/langfuse/save-prompt")
async def save_prompt_to_langfuse(request: SavePromptRequest):
    """Save an optimized prompt to Langfuse as a new version"""
    try:
        payload = {
            "name": request.name,
            "prompt": request.prompt,
            "type": "text",
            "labels": request.labels,
        }
        if request.commit_message:
            payload["commitMessage"] = request.commit_message

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{request.langfuse.host}/api/public/v2/prompts",
                headers={
                    **get_langfuse_auth_header(request.langfuse),
                    "Content-Type": "application/json"
                },
                json=payload
            )
            if response.status_code >= 400:
                error_text = response.text
                raise HTTPException(status_code=response.status_code, detail=f"Langfuse error: {error_text}")
            return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prompt: {str(e)}")


# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
