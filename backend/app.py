import asyncio
import json
import uuid
import threading
import httpx
import os
from typing import Optional, Any, List
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
import litellm
import gepa
from gepa import GEPAAdapter, EvaluationBatch
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database and Auth imports
from .database import (
    init_db, get_db, User, UserSettings, EvolutionSessionDB, SavedPrompt,
    get_user_settings, get_user_by_id
)
from .auth import (
    get_current_user, get_current_admin, get_current_user_optional,
    register_user, authenticate_user, create_access_token, decode_token,
    UserCreate, UserLogin, Token, hash_password
)

app = FastAPI(title="GEPA Prompt Evolution")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("[Startup] Database initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for evolution sessions (keyed by user_id:session_id)
evolution_sessions = {}

def get_user_session_key(user_id: int, session_id: str) -> str:
    """Generate a unique key for user-specific session storage"""
    return f"{user_id}:{session_id}"


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
    output_type: str = "text"  # "text" or "image" - what the prompt generates
    image_model: Optional[ModelConfig] = None  # Model for image generation (DALL-E, etc.)


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
        self.template_variables = []  # List of variable names found in original prompt
        self.variable_sections = []  # Extracted variable lines to re-attach after evolution
        self.seed_prompt_with_markers = None  # Seed prompt with markers for evolution
        self.original_seed_prompt = None  # Original seed prompt before any processing
        self.generated_images = []  # Store generated images for display
        self.output_type = "text"  # "text" or "image"
        self.reflection_model = None  # Model config for retemplatization
        self.retemplatized_candidate = None  # LLM-retemplatized best candidate

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


# ===== TEMPLATE VARIABLE PRESERVATION =====
# Extract variable lines before evolution, re-attach after

MARKER_PREFIX = "[[TEMPLATE_VAR:"
MARKER_SUFFIX = "]]"


def extract_variable_sections(template: str) -> tuple[str, list[dict]]:
    """Extract INPUT DATA lines containing {{variables}} from the template.

    Strategy: Only extract lines that are pure data injection points (like "{{varName}}" alone
    or "Key: {{varName}}"). Leave variables inside JSON examples or complex structures intact.

    Returns:
        tuple: (template_without_extracted_vars, list of {var_name, full_line, position})
    """
    import re

    lines = template.split('\n')
    variable_sections = []
    clean_lines = []

    # Track if we're inside a JSON/code block (don't extract from these)
    in_json_example = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect JSON example blocks
        if stripped.startswith('{') or stripped.startswith('[') or '"scenarios"' in stripped:
            in_json_example = True
        if stripped == '}' or stripped == '}]' or stripped == ']':
            in_json_example = False

        vars_in_line = re.findall(r'\{\{(\w+)\}\}', line)

        if vars_in_line and not in_json_example:
            # Check if this is a "pure data" line vs part of JSON examples
            # Pure data lines: standalone variable, or simple "Label: {{var}}" format
            is_pure_data_line = (
                # Line is just the variable
                stripped == '{{' + vars_in_line[0] + '}}' or
                # Line is "## SECTION:" followed by variable
                re.match(r'^#+\s+\w+.*\{\{', stripped) or
                # Line is "- Key: {{var}}" or "Key: {{var}}"
                re.match(r'^[-*]?\s*[\w\s]+:\s*\{\{', stripped) or
                # Line contains "= {{var}}" (assignment style)
                '= {{' in line
            )

            # Don't extract if line looks like it's part of JSON example
            looks_like_json = (
                '"' in line and ':' in line and ('{{' in line) or
                'step' in line.lower() or
                'type' in line.lower() and '"' in line or
                'url' in line.lower() and '"' in line
            )

            if is_pure_data_line and not looks_like_json:
                # This line contains variables - extract it
                for var in vars_in_line:
                    variable_sections.append({
                        'var_name': var,
                        'full_line': line,
                        'line_index': i,
                        'position': 'end' if i > len(lines) // 2 else 'start'
                    })
                # Don't include this line in the clean version
            else:
                # Keep the line (it's part of examples/structure)
                clean_lines.append(line)
        else:
            clean_lines.append(line)

    clean_template = '\n'.join(clean_lines)

    if variable_sections:
        print(f"[Template] Extracted {len(variable_sections)} pure data variable sections")
        for vs in variable_sections[:5]:  # Only show first 5
            print(f"  - {vs['var_name']}: '{vs['full_line'][:50]}...'")
        if len(variable_sections) > 5:
            print(f"  ... and {len(variable_sections) - 5} more")

    return clean_template, variable_sections


def reattach_variable_sections(evolved_prompt: str, variable_sections: list[dict]) -> str:
    """Re-attach extracted variable sections to the evolved prompt.

    Appends variable lines at the end (most common pattern for context variables).
    """
    if not variable_sections:
        return evolved_prompt

    result = evolved_prompt.rstrip()

    # Group by unique full_line to avoid duplicates
    unique_lines = []
    seen = set()
    for vs in variable_sections:
        line = vs['full_line'].strip()  # Remove extra whitespace
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    # Append variable lines at the end with clear section marker
    if unique_lines:
        result += '\n\n## INPUT DATA\n'
        for line in unique_lines:
            result += line + '\n'

    print(f"[Template] Re-attached {len(unique_lines)} variable lines:")
    for line in unique_lines:
        print(f"  > {line[:80]}...")

    return result


def template_vars_to_markers(template: str) -> tuple[str, list[str]]:
    """Convert {{variableName}} to [[TEMPLATE_VAR:variableName]] markers.

    Returns:
        tuple: (template with markers, list of variable names found)
    """
    import re

    variables = []
    result = template

    # Find all {{variableName}} patterns
    for match in re.finditer(r'\{\{(\w+)\}\}', template):
        var_name = match.group(1)
        variables.append(var_name)
        # Replace with marker
        result = result.replace('{{' + var_name + '}}', f'{MARKER_PREFIX}{var_name}{MARKER_SUFFIX}')

    if variables:
        print(f"[Template] Converted variables to markers: {variables}")

    return result, variables


def markers_to_template_vars(text: str) -> str:
    """Convert [[TEMPLATE_VAR:variableName]] markers back to {{variableName}}.

    Returns:
        Text with markers converted back to template variables
    """
    import re

    result = text

    # Find all markers and convert back
    pattern = re.escape(MARKER_PREFIX) + r'(\w+)' + re.escape(MARKER_SUFFIX)

    for match in re.finditer(pattern, text):
        var_name = match.group(1)
        marker = f'{MARKER_PREFIX}{var_name}{MARKER_SUFFIX}'
        result = result.replace(marker, '{{' + var_name + '}}')
        print(f"[Template] Restored marker to {{{{var_name}}}}")

    return result


def fill_template_with_markers(template: str, data: str) -> str:
    """Fill template that has markers with sample data.

    Temporarily replaces markers with data for LLM evaluation,
    but the candidate prompt itself keeps the markers.
    """
    import re

    result = template

    # Calculate batchSize from data
    batch_size = max(1, len(re.findall(r'<[^>]+>', data)))
    if batch_size == 0:
        batch_size = len([l for l in data.split('\n') if l.strip()])

    # Replace markers with actual data
    pattern = re.escape(MARKER_PREFIX) + r'(\w+)' + re.escape(MARKER_SUFFIX)

    for match in re.finditer(pattern, template):
        var_name = match.group(1)
        marker = f'{MARKER_PREFIX}{var_name}{MARKER_SUFFIX}'

        if var_name.lower() == 'batchsize':
            result = result.replace(marker, str(batch_size))
        else:
            result = result.replace(marker, data)

    return result


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


def llm_retemplatize_variables(
    original_prompt: str,
    evolved_prompt: str,
    model_config: ModelConfig
) -> str:
    """Use LLM to intelligently re-insert template variables into evolved prompt.

    Takes the original prompt with {{variables}} and the evolved prompt,
    asks LLM to place variables in appropriate locations in the evolved version.

    Special handling for prompts with JSON examples: preserve the exact variable
    placements from the original, only evolving the instructional text.
    """
    import re

    print(f"[Retemplatize] Starting LLM retemplatization...")
    print(f"[Retemplatize] Original prompt length: {len(original_prompt)}")
    print(f"[Retemplatize] Evolved prompt length: {len(evolved_prompt)}")

    # Extract all template variables from original
    variables = re.findall(r'\{\{(\w+)\}\}', original_prompt)
    unique_vars = list(set(variables))
    print(f"[Retemplatize] Found {len(unique_vars)} unique variables: {unique_vars}")
    if not unique_vars:
        print("[Retemplatize] No variables found, returning evolved prompt as-is")
        return evolved_prompt

    # Check if original has JSON examples with variables (like OUTPUT FORMAT sections)
    has_json_examples = (
        '"scenarios"' in original_prompt or
        '"scopeType"' in original_prompt or
        '"intent_label"' in original_prompt or
        ('"url":' in original_prompt and '{{' in original_prompt)
    )

    if has_json_examples:
        print("[Retemplatize] Detected JSON examples with variables - using preservation strategy")

        result = evolved_prompt

        # Strategy: Find each line in original that contains a variable and ensure
        # the evolved version has a similar line with the variable intact

        # Build a map of variable -> original lines containing it
        var_to_original_lines = {}
        for var in unique_vars:
            var_pattern = '{{' + var + '}}'
            matching_lines = []
            for line in original_prompt.split('\n'):
                if var_pattern in line:
                    matching_lines.append(line)
            var_to_original_lines[var] = matching_lines
            print(f"[Retemplatize] Variable {{{{{var}}}}} found in {len(matching_lines)} original lines")

        # For each variable, check if it exists in evolved. If not, try to restore it.
        for var in unique_vars:
            var_pattern = '{{' + var + '}}'
            if var_pattern not in result:
                print(f"[Retemplatize] Variable {{{{{var}}}}} missing from evolved - attempting restoration")

                # Get original lines with this variable
                original_lines = var_to_original_lines.get(var, [])

                for orig_line in original_lines:
                    # Try to find a similar line in evolved (without the variable)
                    # Remove the variable to get the "skeleton" of the line
                    skeleton = orig_line.replace(var_pattern, '').strip()

                    # Look for lines in evolved that match the skeleton pattern
                    if len(skeleton) > 10:  # Only if meaningful content remains
                        # Try exact match first
                        if skeleton in result and var_pattern not in result:
                            result = result.replace(skeleton, orig_line.strip(), 1)
                            print(f"[Retemplatize] Restored line with {{{{{var}}}}} via skeleton match")
                            break

                        # Try finding by key phrases
                        key_phrases = {
                            'stage_count': ['Current stage_count', 'stage_count =', 'Stage Count:'],
                            'remaining_actions_json': ['REMAINING ACTIONS:', '## REMAINING ACTIONS'],
                            'component_map_reference': ['COMPONENT MAP', 'componentId reference'],
                            'workflow_type': ['Workflow Type:'],
                            'state_url': ['State URL:', 'Navigate to', '"url":'],
                            'intent_label': ['Intent:', 'intent_label'],
                            'intent_goal': ['Intent:', 'intent_goal'],
                            'scope_type': ['Scope Type:', 'scopeType'],
                        }

                        phrases_to_try = key_phrases.get(var, [])
                        for phrase in phrases_to_try:
                            if phrase in result:
                                # Find the line in evolved that contains this phrase
                                evolved_lines = result.split('\n')
                                for i, eline in enumerate(evolved_lines):
                                    if phrase in eline and var_pattern not in eline:
                                        # Replace this line with the original line
                                        evolved_lines[i] = orig_line
                                        result = '\n'.join(evolved_lines)
                                        print(f"[Retemplatize] Restored {{{{{var}}}}} via phrase '{phrase}'")
                                        break
                                break

        # Special handling: preserve entire OUTPUT FORMAT section for JSON examples
        output_format_match = re.search(r'(#\s*OUTPUT FORMAT.*?)(?=\n─{10,}|\nIMPORTANT:|$)', original_prompt, re.DOTALL | re.IGNORECASE)
        if output_format_match:
            original_output_section = output_format_match.group(1)
            evolved_output_match = re.search(r'(#\s*OUTPUT FORMAT.*?)(?=\n─{10,}|\nIMPORTANT:|$)', result, re.DOTALL | re.IGNORECASE)
            if evolved_output_match:
                result = result[:evolved_output_match.start()] + original_output_section + result[evolved_output_match.end():]
                print("[Retemplatize] Preserved original OUTPUT FORMAT section")

        # Check which variables are still missing
        result_vars = set(re.findall(r'\{\{(\w+)\}\}', result))
        missing = set(unique_vars) - result_vars

        if missing:
            print(f"[Retemplatize] Still missing variables after restoration: {missing}")

            # Direct injection for critical variables
            for var in list(missing):
                original_lines = var_to_original_lines.get(var, [])
                if original_lines:
                    # Find appropriate place to inject
                    first_original_line = original_lines[0]

                    # Try to find section header in original that precedes this line
                    orig_lines_list = original_prompt.split('\n')
                    for i, line in enumerate(orig_lines_list):
                        if line == first_original_line:
                            # Look for preceding section header
                            for j in range(i-1, max(0, i-10), -1):
                                if orig_lines_list[j].startswith('#') or orig_lines_list[j].startswith('─'):
                                    header = orig_lines_list[j]
                                    if header in result:
                                        # Inject after this header
                                        result = result.replace(header, header + '\n' + first_original_line, 1)
                                        print(f"[Retemplatize] Injected {{{{{var}}}}} after header")
                                        missing.discard(var)
                                        break
                            break

        # Final fallback: add remaining missing vars in dedicated section
        result_vars = set(re.findall(r'\{\{(\w+)\}\}', result))
        still_missing = set(unique_vars) - result_vars

        if still_missing:
            print(f"[Retemplatize] Final missing vars to add: {still_missing}")
            # Insert the original lines directly before OUTPUT FORMAT
            missing_lines = []
            for var in still_missing:
                original_lines = var_to_original_lines.get(var, [])
                if original_lines:
                    missing_lines.extend(original_lines)
                else:
                    missing_lines.append(f"- {var}: {{{{{var}}}}}")

            if missing_lines:
                injection = "\n" + "\n".join(missing_lines) + "\n"
                if '# OUTPUT FORMAT' in result:
                    result = result.replace('# OUTPUT FORMAT', injection + '# OUTPUT FORMAT')
                else:
                    result += injection

        # Final verification
        final_vars = set(re.findall(r'\{\{(\w+)\}\}', result))
        print(f"[Retemplatize] Final variables in result: {final_vars}")
        print(f"[Retemplatize] Required variables: {set(unique_vars)}")

        if set(unique_vars).issubset(final_vars):
            print(f"[Retemplatize] SUCCESS: All {len(unique_vars)} variables present")
            return result
        else:
            still_missing = set(unique_vars) - final_vars
            print(f"[Retemplatize] WARNING: Still missing {still_missing}, returning original")
            # Use the original prompt as base - safest fallback
            return original_prompt

    # For prompts WITHOUT JSON examples, use the standard LLM retemplatization
    var_info = {}
    for var in unique_vars:
        for line in original_prompt.split('\n'):
            if '{{' + var + '}}' in line:
                var_info[var] = {
                    'context': line.strip()[:100],
                    'semantic': _get_semantic_meaning(var)
                }
                break
        if var not in var_info:
            var_info[var] = {
                'context': 'standalone',
                'semantic': _get_semantic_meaning(var)
            }

    var_list = "\n".join([f"{i+1}. {{{{{v}}}}} - {var_info[v]['semantic']}" for i, v in enumerate(unique_vars)])

    meta_prompt = f"""You are a prompt template engineer. Your task is to insert ALL template variables into an evolved prompt.

## CRITICAL REQUIREMENT
You MUST include ALL {len(unique_vars)} variables listed below. Missing ANY variable is a FAILURE.

## VARIABLES THAT MUST BE INCLUDED (all {len(unique_vars)}):
{var_list}

## EVOLVED PROMPT TO MODIFY:
{evolved_prompt}

## YOUR TASK:
1. Take the evolved prompt above
2. Insert EVERY variable from the list at an appropriate location
3. Use section headers to organize variables logically

## OUTPUT FORMAT:
Return the complete prompt with all {len(unique_vars)} variables inserted.
Do NOT include any explanation, just the prompt.
Do NOT wrap in markdown code blocks.

REMEMBER: You MUST include all {len(unique_vars)} variables: {', '.join([f'{{{{{v}}}}}' for v in unique_vars])}"""

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            model_string = get_litellm_model_string(model_config)
            response = litellm.completion(
                model=model_string,
                messages=[{"role": "user", "content": meta_prompt}],
                api_key=model_config.api_key,
                temperature=0.3,
                max_tokens=8000,
            )

            result = response.choices[0].message.content.strip()

            if result.startswith("```"):
                lines = result.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result = "\n".join(lines)

            print(f"[Retemplatize] Attempt {attempt+1}: LLM returned response of length {len(result)}")

            result_vars = set(re.findall(r'\{\{(\w+)\}\}', result))
            original_vars = set(unique_vars)

            if original_vars.issubset(result_vars):
                print(f"[Retemplatize] SUCCESS: All {len(original_vars)} variables present")
                return result
            else:
                missing = original_vars - result_vars
                print(f"[Retemplatize] Attempt {attempt+1} missing variables: {missing}")

                if attempt < max_attempts - 1:
                    meta_prompt = f"""FAILED: Your previous response was missing these variables: {missing}

You MUST include ALL of these variables in your response:
{', '.join([f'{{{{{v}}}}}' for v in unique_vars])}

Here is the prompt again. Add ALL missing variables:

{result}

Return the complete prompt with ALL {len(unique_vars)} variables."""
                    continue

                result = _insert_missing_variables(result, missing, var_info)
                print(f"[Retemplatize] Inserted {len(missing)} missing variables via fallback")
                return result

        except Exception as e:
            print(f"[Retemplatize] ERROR on attempt {attempt+1}: {e}")
            import traceback
            print(f"[Retemplatize] Traceback: {traceback.format_exc()}")
            if attempt == max_attempts - 1:
                return reattach_variable_sections(evolved_prompt, original_prompt)

    return evolved_prompt


def _get_semantic_meaning(var_name: str) -> str:
    """Get human-readable description of what a variable represents."""
    meanings = {
        'businessContext': 'Business/application context information',
        'testObjective': 'The objective/goal of the test',
        'domData': 'DOM structure data from the page',
        'omniParserData': 'Vision/OCR data from OmniParser',
        'mutationSection': 'Dynamic element mutations/changes',
        'requiredSelectorsSection': 'Required CSS selectors for elements',
        'formGroupsSection': 'Form field groups and their structure',
        'datasetSection': 'Test data/dataset information',
        # SCENARIO GENERATOR variables
        'stage_count': 'Current stage number (0=fresh, >0=already on page)',
        'remaining_actions_json': 'JSON array of actions to convert to scenario',
        'component_map_reference': 'Component ID to description mapping',
        'workflow_type': 'Type of workflow (form_fill, cell_selection, etc)',
        'state_url': 'URL of the page/state being tested',
        'intent_label': 'Label for the test intent (happy_path, etc)',
        'intent_goal': 'Description of what the test aims to achieve',
        'scope_type': 'Scope of the test (page, form, modal, etc)',
    }
    return meanings.get(var_name, f'Data for {var_name}')


def _insert_missing_variables(prompt: str, missing_vars: set, var_info: dict) -> str:
    """Insert missing variables at logical positions in the prompt."""
    import re

    # Define where each variable type should be inserted
    insertion_points = {
        'businessContext': ('## BUSINESS CONTEXT', 0),  # Near top
        'testObjective': ('## TEST OBJECTIVE', 1),
        'domData': ('## DOM DATA', 2),
        'omniParserData': ('## OMNIPARSER DATA', 3),
        'mutationSection': ('## MUTATIONS', 4),
        'requiredSelectorsSection': ('## REQUIRED SELECTORS', 5),
        'formGroupsSection': ('## FORM GROUPS', 6),
        'datasetSection': ('## DATASET', 7),
    }

    # Group missing vars by their ideal position
    sections_to_add = []
    for var in sorted(missing_vars, key=lambda v: insertion_points.get(v, ('', 99))[1]):
        header, _ = insertion_points.get(var, (f'## {var.upper()}', 99))
        sections_to_add.append(f"\n{header}:\n{{{{{var}}}}}\n")

    # Find a good insertion point - after the first major section
    lines = prompt.split('\n')
    insert_idx = 0

    # Look for existing INPUT DATA or similar section
    for i, line in enumerate(lines):
        if any(marker in line.upper() for marker in ['INPUT DATA', 'CONTEXT', 'DOM DATA', 'OMNIPARSER']):
            insert_idx = i
            break

    # If no good spot found, insert after first ## header
    if insert_idx == 0:
        for i, line in enumerate(lines):
            if line.startswith('##') and i > 0:
                insert_idx = i
                break

    # If still no spot, append at end but before any "CRITICAL" or "RULES" section
    if insert_idx == 0:
        for i, line in enumerate(lines):
            if 'CRITICAL' in line.upper() or 'RULES' in line.upper():
                insert_idx = i
                break
        if insert_idx == 0:
            insert_idx = len(lines)

    # Insert the sections
    result_lines = lines[:insert_idx] + [''.join(sections_to_add)] + lines[insert_idx:]
    return '\n'.join(result_lines)


def fill_template(template: str, data: str) -> str:
    """Fill template variables with sample data.

    Supports {{variableName}}, [[TEMPLATE_VAR:variableName]], and {variableName} syntax.
    If data is a JSON object with keys matching variable names, fills each variable with its value.
    Otherwise, replaces all variables with the full data string.
    """
    import re

    # Find all template variables - support {{var}}, markers, and {var}
    double_brace_vars = re.findall(r'\{\{(\w+)\}\}', template)
    marker_vars = re.findall(re.escape(MARKER_PREFIX) + r'(\w+)' + re.escape(MARKER_SUFFIX), template)
    single_brace_vars = re.findall(r'\{(\w+)\}', template)

    # Filter out common false positives for single brace (like JSON examples)
    single_brace_vars = [v for v in single_brace_vars if v not in ['id', 'type', 'intent', 'purpose', 'confidence']]

    variables = double_brace_vars or marker_vars or single_brace_vars
    unique_vars = list(set(variables))

    print(f"[DEBUG fill_template] Template preview: {template[:200]}...")
    print(f"[DEBUG fill_template] Unique vars: {unique_vars}")
    print(f"[DEBUG fill_template] Data length: {len(data)}")

    # Try to parse data as JSON object for structured variable filling
    data_dict = None
    if data.strip().startswith('{'):
        try:
            data_dict = json.loads(data)
            if isinstance(data_dict, dict):
                print(f"[DEBUG fill_template] Parsed JSON data with keys: {list(data_dict.keys())}")
            else:
                data_dict = None
        except json.JSONDecodeError:
            data_dict = None

    if variables:
        filled = template

        # Calculate batchSize from data (count HTML-like elements or lines)
        batch_size = max(1, len(re.findall(r'<[^>]+>', data)))  # Count HTML tags
        if batch_size == 0:
            batch_size = len([l for l in data.split('\n') if l.strip()])  # Count non-empty lines

        # Helper to get value for a variable
        def get_var_value(var_name: str) -> str:
            if var_name.lower() == 'batchsize':
                return str(batch_size)
            if data_dict and var_name in data_dict:
                val = data_dict[var_name]
                # Convert non-string values to appropriate string representation
                if isinstance(val, (dict, list)):
                    return json.dumps(val, indent=2)
                return str(val)
            # Fallback: use full data string
            return data

        # Fill {{var}} syntax
        for var in double_brace_vars:
            filled = filled.replace('{{' + var + '}}', get_var_value(var))

        # Fill [[TEMPLATE_VAR:var]] markers
        for var in marker_vars:
            marker = f'{MARKER_PREFIX}{var}{MARKER_SUFFIX}'
            filled = filled.replace(marker, get_var_value(var))

        # Fill {var} syntax
        for var in single_brace_vars:
            if var not in ['id', 'type', 'intent', 'purpose', 'confidence']:
                filled = filled.replace('{' + var + '}', get_var_value(var))

        print(f"[DEBUG fill_template] Batch size calculated: {batch_size}")
        print(f"[DEBUG fill_template] After filling (first 500 chars): {filled[:500]}...")
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


def generate_image_sync(model_config: ModelConfig, prompt: str, user_input: str = None) -> dict:
    """Generate an image using Replicate API.

    Supports: Flux, Stable Diffusion XL, Google Imagen, and more.
    Returns dict with 'url' of the generated image.
    """
    import replicate
    import os
    import time

    # Fill template if we have input data
    if user_input:
        final_prompt = fill_template(prompt, user_input)
    else:
        final_prompt = prompt

    print(f"[Image Gen] Generating image with model: {model_config.model}")
    print(f"[Image Gen] Prompt: {final_prompt[:100]}...")

    # Set Replicate API token
    os.environ["REPLICATE_API_TOKEN"] = model_config.api_key

    # Model mapping for Replicate - using verified working models
    replicate_models = {
        # Flux models (Black Forest Labs) - Most reliable
        "flux-1.1-pro": "black-forest-labs/flux-1.1-pro",
        "flux-1.1-pro-ultra": "black-forest-labs/flux-1.1-pro-ultra",
        "flux-schnell": "black-forest-labs/flux-schnell",
        "flux-dev": "black-forest-labs/flux-dev",
        "flux-pro": "black-forest-labs/flux-pro",
        # Stable Diffusion models
        "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "sd-3": "stability-ai/stable-diffusion-3",
        "sd-3.5-large": "stability-ai/stable-diffusion-3.5-large",
        "sd-3.5-large-turbo": "stability-ai/stable-diffusion-3.5-large-turbo",
        "sdxl-lightning": "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
        # Ideogram
        "ideogram-v2": "ideogram-ai/ideogram-v2",
        "ideogram-v2-turbo": "ideogram-ai/ideogram-v2-turbo",
        # Recraft
        "recraft-v3": "recraft-ai/recraft-v3",
        "recraft-v3-svg": "recraft-ai/recraft-v3-svg",
        # Playground
        "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
        # Realistic models - Updated to working versions
        "realistic-vision": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "photon": "luma/photon",
        "photon-flash": "luma/photon-flash",
        # Art styles - Updated to working versions
        "dreamshaper": "lucataco/dreamshaper-xl-turbo:0a1710e0187b01a255302738ca0158ff02a22f4638679533e111082f9dd1b615",
        # Midjourney style alternatives (openjourney deprecated)
        "openjourney": "black-forest-labs/flux-schnell",  # Fallback to flux-schnell
        "midjourney": "black-forest-labs/flux-schnell",   # Fallback to flux-schnell
    }

    model_id = replicate_models.get(model_config.model, model_config.model)

    # If model not found in mapping and looks like a bare name, default to flux-schnell
    if model_id == model_config.model and "/" not in model_id:
        print(f"[Image Gen] Unknown model '{model_id}', falling back to flux-schnell")
        model_id = "black-forest-labs/flux-schnell"

    print(f"[Image Gen] Using Replicate model: {model_id}")

    # Build input parameters based on model type
    # Different models have different input schemas
    if "flux" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
        }
    elif "sdxl" in model_id.lower() or "stable-diffusion" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "width": 1024,
            "height": 1024,
        }
    elif "playground" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "width": 1024,
            "height": 1024,
        }
    elif "dreamshaper" in model_id.lower() or "realistic-vision" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 25,
        }
    elif "ideogram" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "aspect_ratio": "1:1",
        }
    elif "recraft" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "size": "1024x1024",
        }
    elif "photon" in model_id.lower():
        model_input = {
            "prompt": final_prompt,
            "aspect_ratio": "1:1",
        }
    else:
        # Default input format
        model_input = {
            "prompt": final_prompt,
        }

    # Retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Run the model
            output = replicate.run(
                model_id,
                input=model_input
            )

            # Handle different output formats
            if isinstance(output, list):
                image_url = str(output[0])
            elif hasattr(output, 'url'):
                image_url = output.url
            else:
                image_url = str(output)

            print(f"[Image Gen] Generated image URL: {image_url[:80]}...")

            return {
                "url": image_url,
                "prompt": final_prompt,
                "model": model_config.model,
                "revised_prompt": None  # Replicate doesn't revise prompts
            }

        except Exception as e:
            error_str = str(e)
            # Check for rate limiting (429)
            if "429" in error_str or "throttled" in error_str.lower() or "rate limit" in error_str.lower():
                wait_time = 10 * (attempt + 1)  # Progressive backoff: 10s, 20s, 30s
                print(f"[Image Gen] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"[Image Gen] Error: {error_str}")
                raise

    # If all retries failed
    raise Exception(f"Failed to generate image after {max_retries} retries due to rate limiting")


def call_vision_llm_sync(model_config: ModelConfig, prompt: str, image_url: str = None, image_base64: str = None) -> str:
    """Call a vision-capable LLM with an image.

    Supports Groq Llama 4 vision models, OpenAI GPT-4V, Anthropic Claude, etc.
    """
    model_string = get_litellm_model_string(model_config)

    # Build message content with image
    content = []

    # Add the image first (required for vision models)
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    elif image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    # Add the text prompt
    content.append({
        "type": "text",
        "text": prompt
    })

    print(f"[Vision LLM] Calling {model_string} with image URL: {image_url[:50] if image_url else 'N/A'}...")

    try:
        response = litellm.completion(
            model=model_string,
            messages=[
                {"role": "user", "content": content}
            ],
            api_key=model_config.api_key,
        )

        if response and response.choices and len(response.choices) > 0:
            result = response.choices[0].message.content
            print(f"[Vision LLM] Response received: {result[:100] if result else 'Empty'}...")
            return result
        else:
            print(f"[Vision LLM] Warning: Empty response from {model_string}")
            return "SCORE: 50\nFEEDBACK: Unable to evaluate image - vision model returned empty response."

    except Exception as e:
        print(f"[Vision LLM] Error calling {model_string}: {str(e)}")
        raise


class LLMJudgeAdapter(GEPAAdapter[TestInput, Trajectory, str]):
    """
    Custom GEPA adapter that uses LLM-as-a-Judge for evaluation.

    - Prompt X (candidate["system_prompt"]) is the prompt being optimized
    - Prompt Y (judge_prompt) evaluates the outputs
    - Z (judge feedback) drives the evolution

    Supports multi-modal: image generation prompts judged by vision models.
    """

    def __init__(
        self,
        task_model: ModelConfig,
        judge_model: ModelConfig,
        judge_prompt: str,
        session: EvolutionSession,
        image_model: ModelConfig = None
    ):
        self.task_model = task_model
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.session = session
        self.image_model = image_model
        self.eval_count = 0

    def evaluate(
        self,
        batch: list[TestInput],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, str]:
        """
        Run Prompt X on inputs, then use Prompt Y (judge) to score.
        Supports both text and image generation prompts.
        """
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        system_prompt = candidate.get("system_prompt", "")
        is_image_mode = self.session.output_type == "image"

        # Track evaluation count (not same as GEPA iterations)
        self.eval_count += 1
        # Don't override current_iteration here - let GEPAProgressLogger handle it

        mode_label = "image generation" if is_image_mode else "text"
        self.session.log(f"Evaluation round {self.eval_count}: testing {mode_label} candidate on {len(batch)} inputs", {
            "prompt_preview": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt,
            "output_type": self.session.output_type
        })

        for item in batch:
            # Check if stopped
            if not self.session.is_running:
                self.session.log("Evolution stopped by user")
                raise StopIteration("Evolution stopped by user")

            try:
                # Re-attach variable sections before filling template
                # (they were extracted before evolution to preserve them)
                prompt_with_vars = system_prompt
                if self.session.variable_sections:
                    prompt_with_vars = reattach_variable_sections(system_prompt, self.session.variable_sections)

                # Fill template with sample data and run
                filled_prompt = fill_template(prompt_with_vars, item.input_text)

                self.session.log(f"Running filled prompt", {
                    "sample_data_preview": item.input_text[:100] + "..." if len(item.input_text) > 100 else item.input_text,
                    "filled_prompt_preview": filled_prompt[:200] + "..." if len(filled_prompt) > 200 else filled_prompt,
                    "mode": "image" if is_image_mode else "text"
                })

                # === IMAGE MODE: Generate image and judge with vision model ===
                if is_image_mode and self.image_model:
                    # Generate image
                    image_result = generate_image_sync(self.image_model, filled_prompt)
                    image_url = image_result.get("url")
                    output = f"[IMAGE: {image_url}]"

                    # Store for display
                    self.session.generated_images.append({
                        "url": image_url,
                        "prompt": filled_prompt,
                        "revised_prompt": image_result.get("revised_prompt"),
                        "eval_round": self.eval_count
                    })

                    revised = image_result.get("revised_prompt") or ""
                    self.session.log(f"Generated image", {
                        "image_url": image_url[:80] + "..." if len(image_url) > 80 else image_url,
                        "revised_prompt": revised[:100] if revised else ""
                    })

                    # Judge using vision model
                    judge_vision_prompt = f"""{self.judge_prompt}

The image was generated from this prompt:
{filled_prompt}

Please evaluate the image quality and provide:
1. A score from 0-100
2. Specific feedback on what could be improved in the prompt to generate a better image

Format your response as:
SCORE: [number]
FEEDBACK: [your detailed feedback]
"""
                    self.session.log(f"Calling vision judge", {
                        "judge_model": f"{self.judge_model.provider}/{self.judge_model.model}",
                        "image_url_preview": image_url[:60] + "..." if len(image_url) > 60 else image_url
                    })

                    judge_response = call_vision_llm_sync(
                        self.judge_model,
                        judge_vision_prompt,
                        image_url=image_url
                    )

                    self.session.log(f"Vision judge response", {
                        "response_preview": judge_response[:200] + "..." if judge_response and len(judge_response) > 200 else judge_response
                    })

                    outputs.append(output)

                # === TEXT MODE: Standard text generation and judging ===
                else:
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

                # Parse score (same for both modes)
                score = 0.5  # default normalized score
                try:
                    if judge_response:
                        for line in judge_response.split('\n'):
                            if line.strip().upper().startswith('SCORE:'):
                                score_str = line.split(':')[1].strip()
                                raw_score = float(score_str.replace('%', ''))
                                score = raw_score / 100.0  # Normalize to 0-1
                                break
                    else:
                        self.session.log("Warning: No judge response received")
                except Exception as parse_err:
                    self.session.log(f"Warning: Could not parse score from judge response: {str(parse_err)}")

                scores.append(score)

                # Update the stored image with its score (for image mode)
                if is_image_mode and self.session.generated_images:
                    # Update the last added image with its score
                    self.session.generated_images[-1]["score"] = round(score * 100, 1)

                self.session.log(f"Test evaluation", {
                    "input": item.input_text[:50] + "..." if item.input_text else "(empty)",
                    "output": output[:100] + "..." if output else "(empty)",
                    "score": round(score * 100, 1),
                    "is_image": is_image_mode
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
    session.log("=== _populate_stopped_results CALLED ===")
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

    # LLM retemplatization for stopped evolutions too
    session.log("=== STOP: Checking LLM retemplatization conditions ===", {
        "has_best": bool(session.best_candidate),
        "has_original": bool(session.original_seed_prompt),
        "has_model": bool(session.reflection_model),
        "num_variables": len(session.template_variables) if session.template_variables else 0,
        "variables": session.template_variables[:3] if session.template_variables else []  # First 3 only
    })

    if session.best_candidate and session.original_seed_prompt and session.reflection_model and session.template_variables:
        session.log("Running LLM retemplatization for stopped evolution...")
        try:
            session.retemplatized_candidate = llm_retemplatize_variables(
                session.original_seed_prompt,
                session.best_candidate,
                session.reflection_model
            )
            session.log("LLM retemplatization completed", {
                "variables_restored": session.template_variables
            })
        except Exception as e:
            session.log(f"LLM retemplatization failed, using fallback: {str(e)}")
            session.retemplatized_candidate = reattach_variable_sections(
                session.best_candidate, session.variable_sections
            )

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

    # Set output type from config
    session.output_type = config.output_type or "text"

    try:
        # Prepare training data
        # For image mode without specific inputs, use empty placeholder
        if config.output_type == "image" and (not config.test_inputs or config.test_inputs == ['']):
            trainset = [TestInput(input_text="", id=0)]
        else:
            trainset = [
                TestInput(input_text=text, id=i)
                for i, text in enumerate(config.test_inputs)
            ]

        # Store original seed prompt and reflection model for LLM retemplatization later
        session.original_seed_prompt = config.seed_prompt
        session.reflection_model = config.reflection_model or config.judge_model

        # STRATEGY: Extract variable lines BEFORE evolution, re-attach AFTER
        # This ensures {{variables}} are preserved even when GEPA restructures the prompt
        clean_template, variable_sections = extract_variable_sections(config.seed_prompt)
        session.variable_sections = variable_sections
        session.template_variables = [vs['var_name'] for vs in variable_sections]

        if variable_sections:
            session.log("Extracted variable sections for preservation", {
                "variables": session.template_variables,
                "lines_extracted": len(variable_sections)
            })

        # Extract and store template mapping for fallback restoration
        if config.test_inputs:
            session.template_mapping = extract_template_mapping(
                config.seed_prompt,
                config.test_inputs[0]
            )

        # Create seed candidate WITHOUT variable lines (they'll be re-attached)
        # But we need to evaluate with variables, so keep original for evaluation
        session.seed_prompt_with_markers = clean_template

        seed_candidate = {
            "system_prompt": clean_template
        }

        # Create custom adapter with LLM-as-Judge (and optional image model)
        adapter = LLMJudgeAdapter(
            task_model=config.task_model,
            judge_model=config.judge_model,
            judge_prompt=config.judge_prompt,
            session=session,
            image_model=config.image_model
        )

        session.log(f"Starting {'image' if session.output_type == 'image' else 'text'} evolution", {
            "output_type": session.output_type,
            "has_image_model": config.image_model is not None
        })

        # Determine reflection model for GEPA
        reflection_model_config = session.reflection_model  # Already set earlier
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

        # LLM retemplatization: intelligently place {{variables}} back into evolved prompt
        session.log("Checking LLM retemplatization conditions", {
            "has_original_seed": bool(session.original_seed_prompt),
            "has_reflection_model": bool(session.reflection_model),
            "template_variables": session.template_variables,
            "num_variables": len(session.template_variables) if session.template_variables else 0
        })
        if session.original_seed_prompt and session.reflection_model and session.template_variables:
            session.log("Running LLM retemplatization to restore variables...")
            try:
                session.retemplatized_candidate = llm_retemplatize_variables(
                    session.original_seed_prompt,
                    session.best_candidate,
                    session.reflection_model
                )
                session.log("LLM retemplatization completed", {
                    "variables_restored": session.template_variables
                })
            except Exception as e:
                session.log(f"LLM retemplatization failed, using fallback: {str(e)}")
                # Fallback to simple reattachment
                session.retemplatized_candidate = reattach_variable_sections(
                    session.best_candidate, session.variable_sections
                )

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


# ==================== AUTH ENDPOINTS ====================

class SettingsUpdate(BaseModel):
    """Schema for updating user settings."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    replicate_api_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    prompt_config: Optional[dict] = None  # UI config (seed_prompt, test_inputs, etc.)


@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user. First user becomes admin."""
    user = register_user(db, user_data.username, user_data.email, user_data.password)

    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})

    return Token(
        access_token=access_token,
        user=user.to_dict()
    )


@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token."""
    user = authenticate_user(db, credentials.username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    return Token(
        access_token=access_token,
        user=user.to_dict()
    )


@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user.to_dict()


@app.get("/api/auth/settings")
async def get_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's settings (API keys masked)."""
    settings = get_user_settings(db, current_user.id)
    if not settings:
        return {}
    return settings.to_dict(include_keys=False)


@app.put("/api/auth/settings")
async def update_settings(
    settings_data: SettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's settings (API keys, Langfuse config)."""
    settings = get_user_settings(db, current_user.id)

    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)

    # Update only provided fields (don't overwrite with None)
    if settings_data.openai_api_key is not None:
        settings.openai_api_key = settings_data.openai_api_key
    if settings_data.anthropic_api_key is not None:
        settings.anthropic_api_key = settings_data.anthropic_api_key
    if settings_data.google_api_key is not None:
        settings.google_api_key = settings_data.google_api_key
    if settings_data.mistral_api_key is not None:
        settings.mistral_api_key = settings_data.mistral_api_key
    if settings_data.groq_api_key is not None:
        settings.groq_api_key = settings_data.groq_api_key
    if settings_data.replicate_api_key is not None:
        settings.replicate_api_key = settings_data.replicate_api_key
    if settings_data.langfuse_host is not None:
        settings.langfuse_host = settings_data.langfuse_host
    if settings_data.langfuse_public_key is not None:
        settings.langfuse_public_key = settings_data.langfuse_public_key
    if settings_data.langfuse_secret_key is not None:
        settings.langfuse_secret_key = settings_data.langfuse_secret_key
    if settings_data.prompt_config is not None:
        settings.prompt_config = settings_data.prompt_config

    db.commit()
    db.refresh(settings)

    return {"message": "Settings updated successfully", "settings": settings.to_dict(include_keys=False)}


# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/users")
async def list_users(
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return [u.to_dict() for u in users]


@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Delete a user (admin only). Cannot delete self."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": f"User {user.username} deleted successfully"}


@app.post("/api/admin/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Enable/disable a user (admin only). Cannot disable self."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot disable yourself")

    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = not user.is_active
    db.commit()

    status = "enabled" if user.is_active else "disabled"
    return {"message": f"User {user.username} {status}", "is_active": user.is_active}


# ==================== EVOLUTION ENDPOINTS ====================

@app.post("/api/evolution/start")
async def start_evolution(config: EvolutionConfig, current_user: User = Depends(get_current_user_optional)):
    """Start a new GEPA evolution session"""
    session = EvolutionSession(config)

    # Get user_id (use 0 for unauthenticated users for backward compatibility)
    user_id = current_user.id if current_user else 0
    session_key = get_user_session_key(user_id, session.id)
    evolution_sessions[session_key] = session

    # Run GEPA in background thread (it's synchronous)
    thread = threading.Thread(target=run_gepa_evolution, args=(session,))
    thread.start()

    return {"session_id": session.id}


@app.get("/api/evolution/{session_id}/status")
async def get_evolution_status(session_id: str, current_user: User = Depends(get_current_user_optional)):
    """Get current status of an evolution session"""
    user_id = current_user.id if current_user else 0
    session_key = get_user_session_key(user_id, session_id)

    if session_key not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_key]
    improvement = (session.best_score - session.initial_score) if session.initial_score else 0

    # Restore template variables in best_candidate
    best_candidate = session.best_candidate
    restored_candidate = None

    if best_candidate:
        # Use LLM-retemplatized version if available (best quality)
        if session.retemplatized_candidate:
            restored_candidate = session.retemplatized_candidate
        # Re-attach extracted variable sections (e.g., "Current context: {{context}}")
        elif session.variable_sections:
            restored_candidate = reattach_variable_sections(best_candidate, session.variable_sections)
        else:
            # Fallback: try marker conversion
            restored_candidate = markers_to_template_vars(best_candidate)

            # If still no vars and we have template mapping, try value-based restoration
            if restored_candidate == best_candidate and session.template_mapping:
                restored_candidate = restore_template_variables(best_candidate, session.template_mapping)

    # Also restore variables in all candidates
    restored_candidates = []
    for c in session.candidates:
        restored_c = dict(c)
        if c.get("prompt"):
            if session.variable_sections:
                restored_c["prompt"] = reattach_variable_sections(c["prompt"], session.variable_sections)
            else:
                restored_c["prompt"] = markers_to_template_vars(c["prompt"])
        restored_candidates.append(restored_c)

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
        "template_variables": session.template_variables,
        "candidates": restored_candidates,
        "output_type": session.output_type,
        "generated_images": session.generated_images,  # For image mode
    }


@app.get("/api/evolution/{session_id}/stream")
async def stream_evolution(session_id: str, token: str = None, db: Session = Depends(get_db)):
    """Stream evolution progress via SSE"""
    # SSE doesn't support headers, so accept token via query param
    user_id = 0
    if token:
        payload = decode_token(token)
        if payload:
            user_id = payload.get("sub", 0)

    session_key = get_user_session_key(user_id, session_id)

    if session_key not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_key]
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

            if best_candidate:
                # Use LLM-retemplatized version if available (best quality)
                if session.retemplatized_candidate:
                    restored_candidate = session.retemplatized_candidate
                # Re-attach extracted variable sections
                elif session.variable_sections:
                    restored_candidate = reattach_variable_sections(best_candidate, session.variable_sections)
                else:
                    # Fallback: try marker conversion
                    restored_candidate = markers_to_template_vars(best_candidate)

                    # If still no vars and we have template mapping, try value-based restoration
                    if restored_candidate == best_candidate and session.template_mapping:
                        restored_candidate = restore_template_variables(best_candidate, session.template_mapping)

            # Also restore variables in all candidates
            restored_candidates = []
            for c in session.candidates:
                restored_c = dict(c)
                if c.get("prompt"):
                    if session.variable_sections:
                        restored_c["prompt"] = reattach_variable_sections(c["prompt"], session.variable_sections)
                    else:
                        restored_c["prompt"] = markers_to_template_vars(c["prompt"])
                restored_candidates.append(restored_c)

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
                    "template_variables": session.template_variables,
                    "candidates": restored_candidates,
                    "output_type": session.output_type,
                    "generated_images": session.generated_images,
                })
            }

            # Only break when fully done (not "stopping" - wait for thread to finish)
            if session.status in ["completed", "error", "stopped"]:
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post("/api/evolution/{session_id}/stop")
async def stop_evolution(session_id: str, current_user: User = Depends(get_current_user_optional)):
    """Stop an evolution session"""
    user_id = current_user.id if current_user else 0
    session_key = get_user_session_key(user_id, session_id)

    if session_key not in evolution_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = evolution_sessions[session_key]
    session.is_running = False
    # Don't set status to "stopped" here - let the GEPA thread do it
    # after it finishes processing (including LLM retemplatization)
    session.status = "stopping"  # Intermediate status
    session.log("Stop requested by user")
    return {"message": "Stop requested"}


@app.get("/api/providers")
async def get_providers():
    """Get list of supported model providers including vision and image generation models"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "vision_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "supports_vision": True
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "vision_models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "supports_vision": True
            },
            {
                "id": "google",
                "name": "Google",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
                "vision_models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                "supports_vision": True
            },
            {
                "id": "mistral",
                "name": "Mistral",
                "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
                "supports_vision": False
            },
            {
                "id": "groq",
                "name": "Groq",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "gemma2-9b-it"],
                "vision_models": ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"],
                "supports_vision": True
            }
        ],
        "image_providers": [
            {
                "id": "replicate",
                "name": "Replicate",
                "models": [
                    {"id": "flux-1.1-pro", "name": "Flux 1.1 Pro (Best quality)"},
                    {"id": "flux-schnell", "name": "Flux Schnell (Fast)"},
                    {"id": "flux-dev", "name": "Flux Dev"},
                    {"id": "sdxl", "name": "Stable Diffusion XL"},
                    {"id": "sd-3", "name": "Stable Diffusion 3"},
                    {"id": "imagen-3", "name": "Google Imagen 3"},
                    {"id": "ideogram", "name": "Ideogram v2"},
                    {"id": "recraft-v3", "name": "Recraft v3"}
                ]
            }
        ]
    }


# ===== GENERATE TEST INPUTS =====

class GenerateTestInputsRequest(BaseModel):
    seed_prompt: str
    num_inputs: int = 5
    model: ModelConfig
    additional_instructions: Optional[str] = None


@app.post("/api/generate-test-inputs")
async def generate_test_inputs(request: GenerateTestInputsRequest):
    """Generate relevant sample data based on the prompt template"""

    # Detect template variables like {{variableName}}
    import re
    variables = list(set(re.findall(r'\{\{(\w+)\}\}', request.seed_prompt)))

    # Add user's additional instructions if provided
    user_instructions = ""
    if request.additional_instructions:
        user_instructions = f"\n\nADDITIONAL USER INSTRUCTIONS:\n{request.additional_instructions}"

    # Use different strategy based on number of template variables
    if len(variables) > 3:
        # Multiple variables: generate structured JSON objects with all variable values
        variables_schema = "\n".join([f'  - {{{{{v}}}}}' for v in variables])

        # Detect variable types from naming conventions
        type_hints = []
        for v in variables:
            v_lower = v.lower()
            if 'count' in v_lower or 'size' in v_lower or 'num' in v_lower or 'stage' in v_lower:
                type_hints.append(f'  - {v}: INTEGER (use actual number like 0, 1, 2 - NOT a string)')
            elif 'json' in v_lower or 'actions' in v_lower or 'map' in v_lower:
                type_hints.append(f'  - {v}: JSON ARRAY or OBJECT (as actual JSON, not stringified)')
            elif 'url' in v_lower:
                type_hints.append(f'  - {v}: URL string (e.g., "https://example.com/path")')

        type_hints_str = "\n".join(type_hints) if type_hints else ""

        meta_prompt = f"""Analyze this prompt template and generate {request.num_inputs} diverse, realistic test data sets.

The prompt template uses these template variables:
{variables_schema}

PROMPT TEMPLATE:
{request.seed_prompt}
{user_instructions}

IMPORTANT: Generate {request.num_inputs} complete test cases. Each test case must be a JSON object with ALL the template variables as keys.

DATA TYPE REQUIREMENTS:
{type_hints_str}
- Variables with "json" or "actions" in the name should be ACTUAL JSON arrays/objects, NOT stringified JSON
- Variables with "count", "stage", "num", "size" should be INTEGERS (0, 1, 2), NOT strings ("0", "1")
- Variables like "component_map_reference" should be readable multi-line strings, not JSON

For example, if variables are {{{{stage_count}}}}, {{{{remaining_actions_json}}}}, {{{{state_url}}}}, return:
[
  {{
    "stage_count": 0,
    "remaining_actions_json": [
      {{"action": "fill", "target": "Email", "componentId": "input_email_001", "purpose": "Enter email"}}
    ],
    "state_url": "https://app.example.com/login"
  }}
]

RULES:
1. Each object MUST have ALL variables: {', '.join(variables)}
2. Values should be realistic and match what the prompt expects
3. Cover different scenarios (simple, complex, edge cases)
4. Use correct data types: integers for counts, arrays for JSON variables, strings for text
5. Return ONLY a valid JSON array of objects, no markdown, no explanation

Generate the {request.num_inputs} test cases now:"""

    else:
        # Few or no variables: use simpler approach
        variables_hint = ""
        if variables:
            variables_hint = f"\n\nDetected template variables: {', '.join(variables)}. Generate sample data that would fill these variables."

        meta_prompt = f"""Analyze this prompt template and generate {request.num_inputs} diverse, realistic SAMPLE DATA sets that would be used with it.

The prompt template:
{request.seed_prompt}
{variables_hint}{user_instructions}

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
            temperature=0.7,  # Add some creativity but keep it structured
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        # Extract JSON array from response
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            json_str = match.group()

            # Try parsing as-is first
            try:
                test_inputs = json.loads(json_str)

                # If we have multiple variables and got objects, convert to strings for the UI
                if len(variables) > 3 and test_inputs and isinstance(test_inputs[0], dict):
                    # Return as structured data - each item is a JSON string of the object
                    string_inputs = [json.dumps(item, indent=2) for item in test_inputs]
                    return {"test_inputs": string_inputs, "structured": True, "variables": variables}

                return {"test_inputs": test_inputs}
            except json.JSONDecodeError as e:
                print(f"[generate_test_inputs] JSON parse error: {e}")
                pass

            # Try to fix common JSON issues
            try:
                # Fix unescaped newlines inside strings (but not structural ones)
                # This is tricky - we need to be careful not to break valid JSON
                fixed_json = json_str

                # Try parsing with more lenient approach
                test_inputs = json.loads(fixed_json)
                return {"test_inputs": test_inputs}
            except json.JSONDecodeError:
                pass

            # Try extracting individual JSON objects if it's an array of objects
            try:
                # Find all top-level objects in the array
                object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                objects = re.findall(object_pattern, json_str)
                if objects:
                    parsed_objects = []
                    for obj_str in objects:
                        try:
                            parsed_objects.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            continue
                    if parsed_objects:
                        string_inputs = [json.dumps(item, indent=2) for item in parsed_objects]
                        return {"test_inputs": string_inputs, "structured": True, "variables": variables}
            except Exception:
                pass

            # Last resort: return raw content split into chunks
            return {"test_inputs": [content], "raw": True}
        else:
            return {"test_inputs": [content], "raw": True}

    except Exception as e:
        import traceback
        print(f"[generate_test_inputs] Error: {str(e)}")
        print(f"[generate_test_inputs] Traceback: {traceback.format_exc()}")
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


@app.post("/api/langfuse/prompts/{prompt_name:path}")
async def get_langfuse_prompt(prompt_name: str, config: LangfuseConfig, version: Optional[int] = None, label: Optional[str] = None):
    """Get a specific prompt from Langfuse"""
    import urllib.parse

    try:
        # URL encode the prompt name for the API call
        encoded_name = urllib.parse.quote(prompt_name, safe='')

        async with httpx.AsyncClient() as client:
            # Try different labels in order of preference
            labels_to_try = ["optimized", "latest", "production", None]
            if label:
                labels_to_try.insert(0, label)

            response = None
            last_error = None

            for try_label in labels_to_try:
                params = {}
                if version:
                    params["version"] = version
                if try_label:
                    params["label"] = try_label

                print(f"[Langfuse] Trying to fetch '{prompt_name}' with params: {params}")

                response = await client.get(
                    f"{config.host}/api/public/v2/prompts/{encoded_name}",
                    headers=get_langfuse_auth_header(config),
                    params=params if params else None
                )

                if response.status_code == 200:
                    print(f"[Langfuse] Successfully fetched '{prompt_name}' with label '{try_label}'")
                    return response.json()
                else:
                    last_error = response.text
                    print(f"[Langfuse] Failed with label '{try_label}': {response.status_code}")

            # If all attempts failed, raise the last error
            raise HTTPException(status_code=404, detail=f"Prompt not found. Tried labels: {labels_to_try}. Last error: {last_error}")

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        print(f"[Langfuse] Error fetching prompt '{prompt_name}': {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Langfuse API error: {e.response.text}")
    except Exception as e:
        print(f"[Langfuse] Exception fetching prompt '{prompt_name}': {str(e)}")
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
