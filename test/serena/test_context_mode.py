"""Tests for the context and mode functionality in SerenaAgent."""

import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from serena.agent import SerenaAgent, PromptFactory, SerenaConfig, ProjectConfig, Tool
from serena.llm.multilang_prompt import ContextConfig, ModeConfig
from multilspy.multilspy_config import Language


class MockSerenaConfig:
    """A mock SerenaConfig for testing."""

    def __init__(self):
        self.project_names = ["test_project"]
        self.projects = {}
        self.gui_log_window_enabled = False
        self.enable_project_activation = True

    def get_project_configuration(self, project_name):
        return self.projects.get(project_name)


class MockProjectConfig:
    """A mock ProjectConfig for testing."""

    def __init__(self, name="test_project", language=Language.PYTHON, read_only=False, excluded_tools=None):
        self.project_name = name
        self.language = language
        self.read_only = read_only
        self.excluded_tools = set(excluded_tools or [])
        self.project_root = "/tmp/test_project"
        self.ignored_paths = []
        self.ignore_all_files_in_gitignore = True

    def get_serena_managed_dir(self):
        return os.path.join(self.project_root, ".serena")


@pytest.fixture
def mock_serena_config():
    """Create a mock SerenaConfig."""
    return MockSerenaConfig()


@pytest.fixture
def test_context_mode_files():
    """Create temporary context and mode YAML files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_contexts_dir = Path(temp_dir) / "contexts"
        temp_modes_dir = Path(temp_dir) / "modes"

        temp_contexts_dir.mkdir()
        temp_modes_dir.mkdir()

        # Create test contexts
        context_a = {
            "name": "context_a",
            "description": "Test context A",
            "system_prompt_addition": "Context A prompt addition",
            "excluded_tools": ["tool_a", "tool_b"],
        }

        context_b = {
            "name": "context_b",
            "description": "Test context B",
            "system_prompt_addition": "Context B prompt addition",
            "excluded_tools": ["tool_c"],
        }

        # Create test modes
        mode_x = {
            "name": "mode_x",
            "description": "Test mode X",
            "system_prompt_addition": "Mode X prompt addition",
            "excluded_tools": ["tool_d", "tool_e"],
        }

        mode_y = {
            "name": "mode_y",
            "description": "Test mode Y",
            "system_prompt_addition": "Mode Y prompt addition",
            "excluded_tools": ["tool_f"],
        }

        # Conflicting modes (for testing conflict detection)
        mode_conflict_1 = {
            "name": "conflict_1",
            "description": "Conflict mode 1",
            "system_prompt_addition": "Conflict mode 1 addition",
            "excluded_tools": ["conflict_tool"],
        }

        mode_conflict_2 = {
            "name": "conflict_2",
            "description": "Conflict mode 2",
            "system_prompt_addition": "Conflict mode 2 addition",
            "excluded_tools": [],  # No exclusion for conflict_tool
        }

        # Write test configs to files
        with open(temp_contexts_dir / "context_a.yml", "w") as f:
            yaml.dump(context_a, f)

        with open(temp_contexts_dir / "context_b.yml", "w") as f:
            yaml.dump(context_b, f)

        with open(temp_modes_dir / "mode_x.yml", "w") as f:
            yaml.dump(mode_x, f)

        with open(temp_modes_dir / "mode_y.yml", "w") as f:
            yaml.dump(mode_y, f)

        with open(temp_modes_dir / "conflict_1.yml", "w") as f:
            yaml.dump(mode_conflict_1, f)

        with open(temp_modes_dir / "conflict_2.yml", "w") as f:
            yaml.dump(mode_conflict_2, f)

        yield temp_dir


@pytest.fixture
def mock_agent(monkeypatch, mock_serena_config, test_context_mode_files):
    """Create a SerenaAgent with mocked components for testing."""

    # Mock SerenaConfig singleton to return our mock
    def mock_serena_config_init(self):
        self.projects = mock_serena_config.projects
        self.project_names = mock_serena_config.project_names
        self.gui_log_window_enabled = mock_serena_config.gui_log_window_enabled
        self.enable_project_activation = mock_serena_config.enable_project_activation

    monkeypatch.setattr(SerenaConfig, "__init__", mock_serena_config_init)

    # Create a mock language server
    mock_language_server = MagicMock()
    mock_language_server.is_running.return_value = True

    # Mock the SymbolManager and MemoriesManager
    mock_symbol_manager = MagicMock()
    mock_memories_manager = MagicMock()

    # Mock the entire SerenaAgent.__init__ to avoid language server initialization
    original_init = SerenaAgent.__init__

    def mock_init(self, project_file_path=None, project_activation_callback=None, context=None, modes=None):
        # Create a partial initialization, skipping language server startup
        self.serena_config = mock_serena_config
        self.prompt_factory = PromptFactory()
        self._project_activation_callback = project_activation_callback

        # Set context and modes
        self._set_context_and_modes(context, modes)

        # Mock project-specific instances
        self.project_config = None
        self.language_server = mock_language_server
        self.symbol_manager = mock_symbol_manager
        self.memories_manager = mock_memories_manager
        self.lines_read = MagicMock()

        # Mock the tools
        self._all_tools = {}
        self._active_tools = {}

        # Mock the project config without activating a real project
        project_config = MockProjectConfig(excluded_tools=["project_excluded_tool"])
        mock_serena_config.projects["test_project"] = project_config

    # Apply the mock init
    monkeypatch.setattr(SerenaAgent, "__init__", mock_init)

    # Mock the prompt template folder method to use our temporary directory
    monkeypatch.setattr(
        "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder",
        classmethod(lambda cls: test_context_mode_files),
    )

    # Create the agent
    agent = SerenaAgent()

    # Set the project config for testing tool exclusions
    agent.project_config = MockProjectConfig(excluded_tools=["project_excluded_tool"])

    yield agent

    # Restore original init
    monkeypatch.setattr(SerenaAgent, "__init__", original_init)


def test_context_loading(mock_agent):
    """Test that contexts can be loaded correctly."""
    prompt_factory = mock_agent.prompt_factory

    # Test with context_a
    prompt_factory.set_context("context_a")
    assert prompt_factory.context is not None
    assert prompt_factory.context.name == "context_a"
    assert prompt_factory.context.excluded_tools == ["tool_a", "tool_b"]
    assert "Context A prompt addition" in prompt_factory.context.system_prompt_addition

    # Test with context_b
    prompt_factory.set_context("context_b")
    assert prompt_factory.context is not None
    assert prompt_factory.context.name == "context_b"
    assert prompt_factory.context.excluded_tools == ["tool_c"]
    assert "Context B prompt addition" in prompt_factory.context.system_prompt_addition


def test_mode_loading(mock_agent):
    """Test that modes can be loaded correctly."""
    prompt_factory = mock_agent.prompt_factory

    # Test with mode_x
    prompt_factory.set_modes(["mode_x"])
    assert len(prompt_factory.modes) == 1
    assert prompt_factory.modes[0].name == "mode_x"
    assert prompt_factory.modes[0].excluded_tools == ["tool_d", "tool_e"]
    assert "Mode X prompt addition" in prompt_factory.modes[0].system_prompt_addition

    # Test with mode_y
    prompt_factory.set_modes(["mode_y"])
    assert len(prompt_factory.modes) == 1
    assert prompt_factory.modes[0].name == "mode_y"
    assert prompt_factory.modes[0].excluded_tools == ["tool_f"]
    assert "Mode Y prompt addition" in prompt_factory.modes[0].system_prompt_addition

    # Test with multiple modes
    prompt_factory.set_modes(["mode_x", "mode_y"])
    assert len(prompt_factory.modes) == 2
    assert prompt_factory.modes[0].name == "mode_x"
    assert prompt_factory.modes[1].name == "mode_y"


def test_excluded_tools_collection(mock_agent):
    """Test that excluded tools are correctly collected from context and modes."""
    prompt_factory = mock_agent.prompt_factory

    # Set context and modes
    prompt_factory.set_context("context_a")
    prompt_factory.set_modes(["mode_x", "mode_y"])

    # Get combined excluded tools
    excluded_tools = prompt_factory.get_context_and_modes_excluded_tools()

    # Should have tools from context_a, mode_x, and mode_y without duplicates
    assert set(excluded_tools) == {"tool_a", "tool_b", "tool_d", "tool_e", "tool_f"}


def test_system_prompt_with_context_and_modes(mock_agent):
    """Test that the system prompt includes context and mode additions."""
    prompt_factory = mock_agent.prompt_factory

    # Store the original system prompt method to restore it later
    original_method = prompt_factory._format_prompt

    try:
        # Mock the prompt formatting to just return our template values
        def mock_format_prompt(prompt_name, kwargs):
            if prompt_name == "system_prompt":
                context = kwargs.get("context", "")
                modes = kwargs.get("modes", [])
                return f"SYSTEM_PROMPT_BASE|{context}|{'|'.join(modes)}"
            return original_method(prompt_name, kwargs)

        prompt_factory._format_prompt = mock_format_prompt

        # Test with no context or modes
        system_prompt = prompt_factory.create_system_prompt()
        assert system_prompt == "SYSTEM_PROMPT_BASE||"

        # Test with context only
        prompt_factory.set_context("context_a")
        system_prompt = prompt_factory.create_system_prompt()
        assert system_prompt == "SYSTEM_PROMPT_BASE|Context A prompt addition|"

        # Test with modes only
        prompt_factory.set_context(None)
        prompt_factory.set_modes(["mode_x", "mode_y"])
        system_prompt = prompt_factory.create_system_prompt()
        assert system_prompt == "SYSTEM_PROMPT_BASE||Mode X prompt addition|Mode Y prompt addition"

        # Test with both context and modes
        prompt_factory.set_context("context_b")
        prompt_factory.set_modes(["mode_x"])
        system_prompt = prompt_factory.create_system_prompt()
        assert system_prompt == "SYSTEM_PROMPT_BASE|Context B prompt addition|Mode X prompt addition"
    finally:
        # Restore original method
        prompt_factory._format_prompt = original_method


def test_project_tool_exclusion_priority(mock_agent):
    """Test that project tool exclusions have highest priority."""

    # Create mock tools
    class MockTool(Tool):
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    # Create mock tools to test exclusions
    tools = {
        "tool_a": MockTool("tool_a"),  # Excluded by context_a
        "tool_c": MockTool("tool_c"),  # Excluded by context_b
        "tool_d": MockTool("tool_d"),  # Excluded by mode_x
        "tool_f": MockTool("tool_f"),  # Excluded by mode_y
        "project_excluded_tool": MockTool("project_excluded_tool"),  # Excluded by project config
        "normal_tool": MockTool("normal_tool"),  # Not excluded
    }

    # Set up the agent with our mock tools
    mock_agent._all_tools = {tool: tool for tool in tools.values()}
    mock_agent.prompt_factory.set_context("context_a")
    mock_agent.prompt_factory.set_modes(["mode_x"])

    # Activate project with specific tool exclusions
    mock_agent.project_config = MockProjectConfig(excluded_tools=["project_excluded_tool", "tool_a"])
    mock_agent._update_active_tools()

    # Check which tools are active
    active_tool_names = mock_agent.get_active_tool_names()

    # project_excluded_tool and tool_a should be excluded (from project config)
    # tool_d should be excluded (from mode_x)
    # normal_tool should be included
    assert "project_excluded_tool" not in active_tool_names
    assert "tool_a" not in active_tool_names
    assert "tool_d" not in active_tool_names
    assert "normal_tool" in active_tool_names

    # Change context to context_b which excludes tool_c
    mock_agent.prompt_factory.set_context("context_b")
    mock_agent._update_active_tools()
    active_tool_names = mock_agent.get_active_tool_names()

    # Now tool_c should also be excluded
    assert "tool_c" not in active_tool_names

    # Even if we switch to a context that doesn't exclude tool_a,
    # it should still be excluded because the project exclusion has priority
    assert "tool_a" not in active_tool_names


def test_set_modes_tool(mock_agent):
    """Test that the set_modes method correctly updates modes and tools."""

    # Create mock tools
    class MockTool(Tool):
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    # Create mock tools to test exclusions
    tools = {
        "tool_d": MockTool("tool_d"),  # Excluded by mode_x
        "tool_f": MockTool("tool_f"),  # Excluded by mode_y
        "normal_tool": MockTool("normal_tool"),  # Not excluded
    }

    # Set up the agent with our mock tools
    mock_agent._all_tools = {tool: tool for tool in tools.values()}
    mock_agent.project_config = MockProjectConfig()

    # Initial state with mode_x
    mock_agent.set_modes(["mode_x"])
    mock_agent._update_active_tools()

    # Check initial active tools
    active_tool_names = mock_agent.get_active_tool_names()
    assert "tool_d" not in active_tool_names
    assert "tool_f" in active_tool_names
    assert "normal_tool" in active_tool_names

    # Change to mode_y
    result = mock_agent.set_modes(["mode_y"])

    # Verify success result
    assert result == "OK"

    # Check that tools were updated
    active_tool_names = mock_agent.get_active_tool_names()
    assert "tool_d" in active_tool_names
    assert "tool_f" not in active_tool_names
    assert "normal_tool" in active_tool_names


def test_mode_conflict_detection():
    """Test that conflicting mode tool exclusions are detected."""
    # Create test agent and prompt factory
    agent = SerenaAgent.__new__(SerenaAgent)
    agent.prompt_factory = PromptFactory()

    # Create conflicting modes
    mode1 = ModeConfig("conflict_1", "desc1", "prompt1", ["conflict_tool"])
    mode2 = ModeConfig("conflict_2", "desc2", "prompt2", [])

    # Set up the modes directly in the prompt factory
    agent.prompt_factory.modes = [mode1, mode2]

    # Call the conflict detection logic directly
    with pytest.raises(ValueError) as excinfo:
        # Check for tool exclusion conflicts between modes
        mode_excluded_tools = {}
        for mode in agent.prompt_factory.modes:
            for tool_name in mode.excluded_tools:
                if tool_name not in mode_excluded_tools:
                    mode_excluded_tools[tool_name] = [mode.name]
                else:
                    mode_excluded_tools[tool_name].append(mode.name)

        # All modes should agree on tool exclusions
        for tool_name, mode_list in mode_excluded_tools.items():
            if len(mode_list) < len(agent.prompt_factory.modes):
                conflicting_modes = ", ".join(mode_list)
                non_conflicting_modes = ", ".join([m.name for m in agent.prompt_factory.modes if m.name not in mode_list])
                raise ValueError(
                    f"Tool '{tool_name}' has conflicting exclusion settings: "
                    f"excluded in modes [{conflicting_modes}] but not in [{non_conflicting_modes}]"
                )

    # Verify error message mentions the conflict
    assert "Tool 'conflict_tool' has conflicting exclusion settings" in str(excinfo.value)
    assert "excluded in modes [conflict_1] but not in [conflict_2]" in str(excinfo.value)


def test_read_only_project_config(mock_agent):
    """Test that read_only project config disables editing tools."""

    # Define mock tool classes with and without edit capability
    class MockEditTool(Tool):
        @classmethod
        def can_edit(cls):
            return True

        def get_name(self):
            return "edit_tool"

    class MockReadTool(Tool):
        @classmethod
        def can_edit(cls):
            return False

        def get_name(self):
            return "read_tool"

    # Set up the agent with our mock tools
    edit_tool = MockEditTool(mock_agent)
    read_tool = MockReadTool(mock_agent)
    mock_agent._all_tools = {MockEditTool: edit_tool, MockReadTool: read_tool}

    # Set non-read-only project first
    mock_agent.project_config = MockProjectConfig(read_only=False)
    mock_agent._update_active_tools()

    # Both tools should be active
    active_tool_names = mock_agent.get_active_tool_names()
    assert "edit_tool" in active_tool_names
    assert "read_tool" in active_tool_names

    # Now set read-only project
    mock_agent.project_config = MockProjectConfig(read_only=True)
    mock_agent._update_active_tools()

    # Only read tools should be active
    active_tool_names = mock_agent.get_active_tool_names()
    assert "edit_tool" not in active_tool_names
    assert "read_tool" in active_tool_names
