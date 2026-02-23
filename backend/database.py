"""
Database models and configuration for GEPA Prompt Evolution.
Uses SQLite with SQLAlchemy ORM.
"""

import os
import json
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from cryptography.fernet import Fernet

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gepa_evolution.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Encryption key for API keys - generate if not exists
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate and save to .env file if not exists
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"ENCRYPTION_KEY={ENCRYPTION_KEY}\n")
            f.write("JWT_SECRET_KEY=your-secret-key-change-in-production\n")

cipher = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def encrypt_value(value: str) -> str:
    """Encrypt a string value."""
    if not value:
        return value
    return cipher.encrypt(value.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt an encrypted string value."""
    if not encrypted:
        return encrypted
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except Exception:
        # Return as-is if decryption fails (might be unencrypted)
        return encrypted


# ==================== MODELS ====================

class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")
    evolution_sessions = relationship("EvolutionSessionDB", back_populates="user", cascade="all, delete-orphan")
    saved_prompts = relationship("SavedPrompt", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class UserSettings(Base):
    """User settings including API keys (encrypted)."""
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # LLM Provider API Keys (stored encrypted)
    _openai_api_key = Column("openai_api_key", Text)
    _anthropic_api_key = Column("anthropic_api_key", Text)
    _google_api_key = Column("google_api_key", Text)
    _mistral_api_key = Column("mistral_api_key", Text)
    _groq_api_key = Column("groq_api_key", Text)
    _replicate_api_key = Column("replicate_api_key", Text)

    # Langfuse Config (stored encrypted)
    langfuse_host = Column(String(255))
    _langfuse_public_key = Column("langfuse_public_key", Text)
    _langfuse_secret_key = Column("langfuse_secret_key", Text)

    # LangSmith Config (stored encrypted)
    _langsmith_api_key = Column("langsmith_api_key", Text)
    langsmith_project = Column(String(255))

    # UI prompt config (JSON - stores seed_prompt, test_inputs, model selections, etc.)
    prompt_config = Column(JSON, default=dict)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="settings")

    # Encrypted property getters/setters
    @property
    def openai_api_key(self):
        return decrypt_value(self._openai_api_key) if self._openai_api_key else None

    @openai_api_key.setter
    def openai_api_key(self, value):
        self._openai_api_key = encrypt_value(value) if value else None

    @property
    def anthropic_api_key(self):
        return decrypt_value(self._anthropic_api_key) if self._anthropic_api_key else None

    @anthropic_api_key.setter
    def anthropic_api_key(self, value):
        self._anthropic_api_key = encrypt_value(value) if value else None

    @property
    def google_api_key(self):
        return decrypt_value(self._google_api_key) if self._google_api_key else None

    @google_api_key.setter
    def google_api_key(self, value):
        self._google_api_key = encrypt_value(value) if value else None

    @property
    def mistral_api_key(self):
        return decrypt_value(self._mistral_api_key) if self._mistral_api_key else None

    @mistral_api_key.setter
    def mistral_api_key(self, value):
        self._mistral_api_key = encrypt_value(value) if value else None

    @property
    def groq_api_key(self):
        return decrypt_value(self._groq_api_key) if self._groq_api_key else None

    @groq_api_key.setter
    def groq_api_key(self, value):
        self._groq_api_key = encrypt_value(value) if value else None

    @property
    def replicate_api_key(self):
        return decrypt_value(self._replicate_api_key) if self._replicate_api_key else None

    @replicate_api_key.setter
    def replicate_api_key(self, value):
        self._replicate_api_key = encrypt_value(value) if value else None

    @property
    def langfuse_public_key(self):
        return decrypt_value(self._langfuse_public_key) if self._langfuse_public_key else None

    @langfuse_public_key.setter
    def langfuse_public_key(self, value):
        self._langfuse_public_key = encrypt_value(value) if value else None

    @property
    def langfuse_secret_key(self):
        return decrypt_value(self._langfuse_secret_key) if self._langfuse_secret_key else None

    @langfuse_secret_key.setter
    def langfuse_secret_key(self, value):
        self._langfuse_secret_key = encrypt_value(value) if value else None

    @property
    def langsmith_api_key(self):
        return decrypt_value(self._langsmith_api_key) if self._langsmith_api_key else None

    @langsmith_api_key.setter
    def langsmith_api_key(self, value):
        self._langsmith_api_key = encrypt_value(value) if value else None

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        provider_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "mistral": self.mistral_api_key,
            "groq": self.groq_api_key,
            "replicate": self.replicate_api_key,
        }
        return provider_map.get(provider.lower())

    def to_dict(self, include_keys=False):
        """Convert to dict. Optionally mask API keys."""
        result = {
            "langfuse_host": self.langfuse_host,
            "prompt_config": self.prompt_config or {},
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

        if include_keys:
            # Return actual keys (for internal use)
            result.update({
                "openai_api_key": self.openai_api_key,
                "anthropic_api_key": self.anthropic_api_key,
                "google_api_key": self.google_api_key,
                "mistral_api_key": self.mistral_api_key,
                "groq_api_key": self.groq_api_key,
                "replicate_api_key": self.replicate_api_key,
                "langfuse_public_key": self.langfuse_public_key,
                "langfuse_secret_key": self.langfuse_secret_key,
                "langsmith_api_key": self.langsmith_api_key,
                "langsmith_project": self.langsmith_project,
            })
        else:
            # Return masked keys (for frontend display)
            def mask_key(key):
                if not key:
                    return None
                if len(key) <= 8:
                    return "****"
                return key[:4] + "****" + key[-4:]

            result.update({
                "openai_api_key": mask_key(self.openai_api_key),
                "anthropic_api_key": mask_key(self.anthropic_api_key),
                "google_api_key": mask_key(self.google_api_key),
                "mistral_api_key": mask_key(self.mistral_api_key),
                "groq_api_key": mask_key(self.groq_api_key),
                "replicate_api_key": mask_key(self.replicate_api_key),
                "langfuse_public_key": mask_key(self.langfuse_public_key),
                "langfuse_secret_key": mask_key(self.langfuse_secret_key),
                "langsmith_api_key": mask_key(self.langsmith_api_key),
                "langsmith_project": self.langsmith_project,
            })

        return result


class EvolutionSessionDB(Base):
    """Evolution session stored in database."""
    __tablename__ = "evolution_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Config (stored as JSON)
    config = Column(JSON)

    # Status
    status = Column(String(20), default="pending")
    current_iteration = Column(Integer, default=0)
    max_iterations = Column(Integer)

    # Scores
    initial_score = Column(Float)
    best_score = Column(Float)

    # Results
    seed_prompt = Column(Text)
    best_candidate = Column(Text)
    retemplatized_candidate = Column(Text)
    candidates = Column(JSON)  # Array of {prompt, score}

    # Template variables
    template_variables = Column(JSON)
    variable_sections = Column(JSON)

    # Logs & Images
    logs = Column(JSON)
    generated_images = Column(JSON)
    output_type = Column(String(10), default="text")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationship
    user = relationship("User", back_populates="evolution_sessions")

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "initial_score": self.initial_score,
            "best_score": self.best_score,
            "seed_prompt": self.seed_prompt,
            "best_candidate": self.best_candidate,
            "retemplatized_candidate": self.retemplatized_candidate,
            "candidates": self.candidates or [],
            "template_variables": self.template_variables or [],
            "variable_sections": self.variable_sections or [],
            "logs": self.logs or [],
            "generated_images": self.generated_images or [],
            "output_type": self.output_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class SavedPrompt(Base):
    """Saved prompts with sample data per user."""
    __tablename__ = "saved_prompts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    prompt_content = Column(Text, nullable=False)
    judge_prompt = Column(Text)
    sample_data = Column(JSON)  # Test inputs
    additional_instructions = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="saved_prompts")

    # Unique constraint on user_id + name
    __table_args__ = (
        # Removed UniqueConstraint - will handle in application logic
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "prompt_content": self.prompt_content,
            "judge_prompt": self.judge_prompt,
            "sample_data": self.sample_data,
            "additional_instructions": self.additional_instructions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ReportedFailure(Base):
    """Reported failures from production for trace-driven evolution."""
    __tablename__ = "reported_failures"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Which prompt failed
    prompt_name = Column(String(255), nullable=False, index=True)

    # The failure data
    input_text = Column(Text, nullable=False)  # User's original request
    output_text = Column(Text, nullable=False)  # LLM's bad response
    expected_output = Column(Text)  # What user wanted (optional correction)

    # Categorization
    failure_category = Column(String(50))  # WRONG_FORMAT, INCOMPLETE, HALLUCINATION, etc.

    # Source metadata
    trace_id = Column(String(100))  # Langfuse/LangSmith trace ID if available
    trace_source = Column(String(20))  # "langfuse", "langsmith", "manual"
    session_id = Column(String(100))  # Chat session ID
    extra_metadata = Column(JSON)  # Additional context

    # Status
    used_in_evolution = Column(Boolean, default=False)  # Whether used in an evolution run
    evolution_session_id = Column(String(36))  # Which evolution used this

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User")

    def to_dict(self):
        return {
            "id": self.id,
            "prompt_name": self.prompt_name,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "expected_output": self.expected_output,
            "failure_category": self.failure_category,
            "trace_id": self.trace_id,
            "trace_source": self.trace_source,
            "session_id": self.session_id,
            "extra_metadata": self.extra_metadata,
            "used_in_evolution": self.used_in_evolution,
            "evolution_session_id": self.evolution_session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TraceEvolutionSession(Base):
    """Evolution session driven by trace data (Langfuse/LangSmith/Manual)."""
    __tablename__ = "trace_evolution_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Link to base evolution session
    evolution_session_id = Column(String(36), ForeignKey("evolution_sessions.id"))

    # Trace source configuration
    trace_source = Column(String(20))  # "langfuse", "langsmith", "manual"
    prompt_name = Column(String(255))  # Prompt being evolved
    trace_filters = Column(JSON)  # Stored filter configuration

    # Trace analysis results
    total_traces_analyzed = Column(Integer)
    failing_traces_count = Column(Integer)
    failure_patterns = Column(JSON)  # Detected patterns from LLM analysis

    # Trace samples used
    trace_samples = Column(JSON)  # List of TraceSample dicts
    trace_ids = Column(JSON)  # List of trace IDs used

    # Judge enhancement
    original_judge_prompt = Column(Text)
    enhanced_judge_prompt = Column(Text)

    # Results tracking
    improvement_by_pattern = Column(JSON)  # {pattern: {before_score, after_score}}

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
    evolution_session = relationship("EvolutionSessionDB")

    def to_dict(self):
        return {
            "id": self.id,
            "evolution_session_id": self.evolution_session_id,
            "trace_source": self.trace_source,
            "prompt_name": self.prompt_name,
            "trace_filters": self.trace_filters,
            "total_traces_analyzed": self.total_traces_analyzed,
            "failing_traces_count": self.failing_traces_count,
            "failure_patterns": self.failure_patterns,
            "trace_samples": self.trace_samples,
            "trace_ids": self.trace_ids,
            "original_judge_prompt": self.original_judge_prompt,
            "enhanced_judge_prompt": self.enhanced_judge_prompt,
            "improvement_by_pattern": self.improvement_by_pattern,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PipelineDefinition(Base):
    """Saved pipeline definitions (DAG of prompt nodes)."""
    __tablename__ = "pipeline_definitions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")

    # Full pipeline structure: nodes, edges, pipeline_inputs, pipeline_output
    pipeline_json = Column(JSON, nullable=False)

    # Separate for easy updates
    judge_prompt = Column(Text, default="")
    test_inputs = Column(JSON, default=list)  # Array of dicts

    # Model configs
    task_model_config = Column(JSON)
    judge_model_config = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pipeline_json": self.pipeline_json,
            "judge_prompt": self.judge_prompt,
            "test_inputs": self.test_inputs,
            "task_model_config": self.task_model_config,
            "judge_model_config": self.judge_model_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PipelineEvolutionSession(Base):
    """Pipeline evolution session tracking."""
    __tablename__ = "pipeline_evolution_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    pipeline_id = Column(String(36), ForeignKey("pipeline_definitions.id"), nullable=False)

    status = Column(String(20), default="pending")  # pending/running/completed/error/stopped

    # Model configs
    task_model_config = Column(JSON)
    judge_model_config = Column(JSON)
    max_iterations = Column(Integer, default=5)
    max_rounds = Column(Integer, default=3)

    # Pipeline snapshots
    original_pipeline_json = Column(JSON)
    evolved_pipeline_json = Column(JSON)

    # Scores
    initial_score = Column(Float)
    best_score = Column(Float)

    # Per-node evolution history
    node_evolution_log = Column(JSON, default=list)

    # Intermediate outputs for debugging
    intermediate_outputs = Column(JSON)

    logs = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    user = relationship("User")
    pipeline = relationship("PipelineDefinition")

    def to_dict(self):
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "initial_score": self.initial_score,
            "best_score": self.best_score,
            "node_evolution_log": self.node_evolution_log,
            "original_pipeline_json": self.original_pipeline_json,
            "evolved_pipeline_json": self.evolved_pipeline_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ==================== DATABASE UTILITIES ====================

def init_db():
    """Initialize database and create tables."""
    Base.metadata.create_all(bind=engine)

    # Run migrations for existing databases
    _run_migrations()

    print("[Database] Tables created successfully")


def _run_migrations():
    """Run simple migrations for SQLite (add missing columns)."""
    from sqlalchemy import text, inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Check if prompt_config column exists in user_settings
    if 'user_settings' in tables:
        columns = [col['name'] for col in inspector.get_columns('user_settings')]
        if 'prompt_config' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE user_settings ADD COLUMN prompt_config TEXT"))
                conn.commit()
                print("[Migration] Added prompt_config column to user_settings")

        # Add LangSmith columns
        if 'langsmith_api_key' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE user_settings ADD COLUMN langsmith_api_key TEXT"))
                conn.commit()
                print("[Migration] Added langsmith_api_key column to user_settings")

        if 'langsmith_project' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE user_settings ADD COLUMN langsmith_project VARCHAR(255)"))
                conn.commit()
                print("[Migration] Added langsmith_project column to user_settings")

    # Create reported_failures table if not exists
    if 'reported_failures' not in tables:
        ReportedFailure.__table__.create(bind=engine)
        print("[Migration] Created reported_failures table")

    # Create trace_evolution_sessions table if not exists
    if 'trace_evolution_sessions' not in tables:
        TraceEvolutionSession.__table__.create(bind=engine)
        print("[Migration] Created trace_evolution_sessions table")

    # Create pipeline tables if not exists
    if 'pipeline_definitions' not in tables:
        PipelineDefinition.__table__.create(bind=engine)
        print("[Migration] Created pipeline_definitions table")

    if 'pipeline_evolution_sessions' not in tables:
        PipelineEvolutionSession.__table__.create(bind=engine)
        print("[Migration] Created pipeline_evolution_sessions table")


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_user(db, username: str, email: str, password_hash: str, is_admin: bool = False) -> User:
    """Create a new user."""
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        is_admin=is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create empty settings for user
    settings = UserSettings(user_id=user.id)
    db.add(settings)
    db.commit()

    return user


def get_user_by_username(db, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def get_user_settings(db, user_id: int) -> Optional[UserSettings]:
    """Get user settings."""
    return db.query(UserSettings).filter(UserSettings.user_id == user_id).first()


def count_users(db) -> int:
    """Count total users in database."""
    return db.query(User).count()
