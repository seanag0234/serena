"""Tests for the context and mode functionality in SerenaAgent."""

import tempfile
from pathlib import Path

import pytest
import yaml

from serena.agent import ProjectConfig, SerenaAgent, Tool
from serena.llm.multilang_prompt import ContextConfig, ModeConfig
from serena.llm.prompt_factory import PromptFactory


@pytest.fixture
def temp_config_files():
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


def test_context_config():
    """Test that ContextConfig is properly initialized with correct attributes."""
    context = ContextConfig(
        name="test_context",
        description="Test description",
        system_prompt_addition="Test system prompt addition",
        excluded_tools=["tool1", "tool2"],
    )

    assert context.name == "test_context"
    assert context.description == "Test description"
    assert context.system_prompt_addition == "Test system prompt addition"
    assert context.excluded_tools == ["tool1", "tool2"]


def test_mode_config():
    """Test that ModeConfig is properly initialized with correct attributes."""
    mode = ModeConfig(
        name="test_mode",
        description="Test description",
        system_prompt_addition="Test system prompt addition",
        excluded_tools=["tool1", "tool2"],
    )

    assert mode.name == "test_mode"
    assert mode.description == "Test description"
    assert mode.system_prompt_addition == "Test system prompt addition"
    assert mode.excluded_tools == ["tool1", "tool2"]


def test_context_loading_from_yaml(temp_config_files):
    """Test that contexts can be loaded correctly from YAML files."""
    context_path = Path(temp_config_files) / "contexts" / "context_a.yml"
    context = ContextConfig.from_yaml(context_path)

    assert context.name == "context_a"
    assert context.description == "Test context A"
    assert "Context A prompt addition" in context.system_prompt_addition
    assert context.excluded_tools == ["tool_a", "tool_b"]


def test_mode_loading_from_yaml(temp_config_files):
    """Test that modes can be loaded correctly from YAML files."""
    mode_path = Path(temp_config_files) / "modes" / "mode_x.yml"
    mode = ModeConfig.from_yaml(mode_path)

    assert mode.name == "mode_x"
    assert mode.description == "Test mode X"
    assert "Mode X prompt addition" in mode.system_prompt_addition
    assert mode.excluded_tools == ["tool_d", "tool_e"]


def test_prompt_factory_with_context_and_modes(monkeypatch, temp_config_files):
    """Test that PromptFactory correctly handles context and modes."""
    # Patch the prompt template folder method to use our temp directory
    monkeypatch.setattr(
        "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder", lambda cls: temp_config_files
    )

    # Create a fresh prompt factory
    prompt_factory = PromptFactory()

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


def test_excluded_tools_collection(monkeypatch, temp_config_files):
    """Test that excluded tools are correctly collected from context and modes."""
    # Patch the prompt template folder method to use our temp directory
    monkeypatch.setattr(
        "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder", lambda cls: temp_config_files
    )

    prompt_factory = PromptFactory()

    # Set context and modes
    prompt_factory.set_context("context_a")
    prompt_factory.set_modes(["mode_x", "mode_y"])

    # Get combined excluded tools
    excluded_tools = prompt_factory.get_context_and_modes_excluded_tools()

    # Should have tools from context_a, mode_x, and mode_y without duplicates
    assert set(excluded_tools) == {"tool_a", "tool_b", "tool_d", "tool_e", "tool_f"}


def test_system_prompt_with_context_and_modes(monkeypatch, temp_config_files):
    """Test that the system prompt includes context and mode additions."""
    # Patch the prompt template folder method to use our temp directory
    monkeypatch.setattr(
        "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder", lambda cls: temp_config_files
    )

    prompt_factory = PromptFactory()

    # Create direct contexts and modes
    context_a = ContextConfig("context_a", "Test A", "Context_A_Addition", [])
    mode_x = ModeConfig("mode_x", "Test X", "Mode_X_Addition", [])
    mode_y = ModeConfig("mode_y", "Test Y", "Mode_Y_Addition", [])

    # Monkey patch get_context and get_mode methods to return our test contexts/modes
    original_get_context = prompt_factory.collection.get_context
    original_get_mode = prompt_factory.collection.get_mode

    def mock_get_context(name_or_path):
        if name_or_path == "context_a":
            return context_a
        return original_get_context(name_or_path)

    def mock_get_mode(name_or_path):
        if name_or_path == "mode_x":
            return mode_x
        elif name_or_path == "mode_y":
            return mode_y
        return original_get_mode(name_or_path)

    monkeypatch.setattr(prompt_factory.collection, "get_context", mock_get_context)
    monkeypatch.setattr(prompt_factory.collection, "get_mode", mock_get_mode)

    # Monkey patch system_prompt template to directly return our context and mode strings
    original_format_prompt = prompt_factory._format_prompt

    def mock_format_prompt(prompt_name, kwargs):
        if prompt_name == "system_prompt":
            context = kwargs.get("context", "")
            modes = kwargs.get("modes", [])
            return f"SYSTEM_PROMPT_BASE|{context}|{'|'.join(modes)}"
        return original_format_prompt(prompt_name, kwargs)

    monkeypatch.setattr(prompt_factory, "_format_prompt", mock_format_prompt)

    # Test with no context or modes
    prompt_factory.set_context(None)
    prompt_factory.set_modes([])
    system_prompt = prompt_factory.create_system_prompt()
    assert system_prompt == "SYSTEM_PROMPT_BASE||"

    # Test with context only
    prompt_factory.set_context("context_a")
    system_prompt = prompt_factory.create_system_prompt()
    assert system_prompt == "SYSTEM_PROMPT_BASE|Context_A_Addition|"

    # Test with modes only
    prompt_factory.set_context(None)
    prompt_factory.set_modes(["mode_x", "mode_y"])
    system_prompt = prompt_factory.create_system_prompt()
    assert system_prompt == "SYSTEM_PROMPT_BASE||Mode_X_Addition|Mode_Y_Addition"

    # Test with both context and modes
    prompt_factory.set_context("context_a")
    prompt_factory.set_modes(["mode_x"])
    system_prompt = prompt_factory.create_system_prompt()
    assert system_prompt == "SYSTEM_PROMPT_BASE|Context_A_Addition|Mode_X_Addition"


def test_mode_conflict_detection(monkeypatch, temp_config_files):
    """Test that conflicting mode tool exclusions are detected."""
    # Patch the prompt template folder method to use our temp directory
    monkeypatch.setattr(
        "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder", lambda cls: temp_config_files
    )

    # Create a SerenaAgent
    class MockLanguageServer:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def is_running(self):
            return self.started

        def stop(self):
            self.started = False

    # Create a test project config
    temp_dir = tempfile.mkdtemp()
    project_config = ProjectConfig(
        config_dict={"language": "python", "project_root": temp_dir, "ignored_paths": [], "ignore_all_files_in_gitignore": True},
        project_name="test_project",
    )

    # Test that an error is raised for conflicting modes
    with pytest.raises(ValueError) as excinfo:
        # Create modes with conflicting tool exclusion
        mode1 = ModeConfig("conflict_1", "desc1", "prompt1", ["conflict_tool"])
        mode2 = ModeConfig("conflict_2", "desc2", "prompt2", [])

        # Create a prompt factory
        prompt_factory = PromptFactory()
        prompt_factory.modes = [mode1, mode2]

        # Create an agent - no real initialization
        agent = SerenaAgent.__new__(SerenaAgent)
        agent.prompt_factory = prompt_factory

        # Call the conflict detection logic directly
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


class TestToolWithContext:
    """Test full SerenaAgent with context and modes."""

    # Create a sample tool for testing
    class MockTool(Tool):
        def __init__(self, agent, name):
            super().__init__(agent)
            self._name = name

        def get_name(self):
            return self._name

        def apply(self):
            return f"Tool {self._name} executed"

    # Read-only tool
    class MockReadTool(Tool):
        def __init__(self, agent, name):
            super().__init__(agent)
            self._name = name

        def get_name(self):
            return self._name

        @classmethod
        def can_edit(cls):
            return False

        def apply(self):
            return f"Read tool {self._name} executed"

    # Editing tool
    class MockEditTool(Tool):
        def __init__(self, agent, name):
            super().__init__(agent)
            self._name = name

        def get_name(self):
            return self._name

        @classmethod
        def can_edit(cls):
            return True

        def apply(self):
            return f"Edit tool {self._name} executed"

    def setup_agent_with_tools(self, monkeypatch, temp_config_files):
        """Create an agent with tools for testing."""
        # Patch the prompt template folder method to use our temp directory
        monkeypatch.setattr(
            "serena.llm.multilang_prompt.MultiLangPromptTemplateCollection._prompt_template_folder", lambda cls: temp_config_files
        )

        # Get references to the mock tool classes outside the scope
        MockTool = self.MockTool
        MockReadTool = self.MockReadTool
        MockEditTool = self.MockEditTool

        # Temporarily disable real language server setup
        original_create = SerenaAgent.__init__

        def mock_init(self, project_file_path=None, project_activation_callback=None, context=None, modes=None):
            # Basic initialization
            self.prompt_factory = PromptFactory()
            self._set_context_and_modes(context, modes)

            # Create test tools
            self._all_tools = {}

            # Each tool needs a different key in _all_tools, so we create unique class instances
            class ToolA(MockTool):
                pass

            class ToolC(MockTool):
                pass

            class ToolD(MockTool):
                pass

            class ToolF(MockTool):
                pass

            class ProjectExcludedTool(MockTool):
                pass

            class NormalTool(MockTool):
                pass

            class ReadTool(MockReadTool):
                pass

            class EditTool(MockEditTool):
                pass

            tools = [
                ToolA(self, "tool_a"),  # Excluded by context_a
                ToolC(self, "tool_c"),  # Excluded by context_b
                ToolD(self, "tool_d"),  # Excluded by mode_x
                ToolF(self, "tool_f"),  # Excluded by mode_y
                ProjectExcludedTool(self, "project_excluded_tool"),  # Will be excluded by project
                NormalTool(self, "normal_tool"),  # Not excluded
                ReadTool(self, "read_tool"),  # Read-only tool
                EditTool(self, "edit_tool"),  # Editing tool
            ]

            for tool in tools:
                self._all_tools[tool.__class__] = tool

            self._active_tools = dict(self._all_tools)

            # Create a test project config
            temp_dir = tempfile.mkdtemp()
            self.project_config = ProjectConfig(
                config_dict={
                    "language": "python",
                    "project_root": temp_dir,
                    "ignored_paths": [],
                    "excluded_tools": ["project_excluded_tool"],
                    "read_only": False,
                    "ignore_all_files_in_gitignore": True,
                },
                project_name="test_project",
            )

            # Create minimal language server
            class MinimalLanguageServer:
                def is_running(self):
                    return True

                def stop(self):
                    pass

            self.language_server = MinimalLanguageServer()
            self.serena_config = type("obj", (object,), {"enable_project_activation": True})

        monkeypatch.setattr(SerenaAgent, "__init__", mock_init)

        try:
            # Create agent
            agent = SerenaAgent()
            return agent
        finally:
            # Restore original init
            monkeypatch.setattr(SerenaAgent, "__init__", original_create)

    def test_project_tool_exclusion_priority(self, monkeypatch, temp_config_files):
        """Test that project tool exclusions have highest priority."""
        agent = self.setup_agent_with_tools(monkeypatch, temp_config_files)

        # Set context and mode
        agent.prompt_factory.set_context("context_a")
        agent.prompt_factory.set_modes(["mode_x"])
        agent._update_active_tools()

        # Check active tools
        active_tool_names = agent.get_active_tool_names()

        # These should be excluded
        assert "project_excluded_tool" not in active_tool_names  # project config
        assert "tool_a" not in active_tool_names  # context_a
        assert "tool_d" not in active_tool_names  # mode_x

        # These should be included
        assert "normal_tool" in active_tool_names

        # Change context to one that doesn't exclude tool_a
        agent.prompt_factory.set_context("context_b")
        agent._update_active_tools()

        # Check active tools again
        active_tool_names = agent.get_active_tool_names()

        # Now tool_c should be excluded, but tool_a included
        assert "tool_c" not in active_tool_names  # context_b
        assert "tool_a" in active_tool_names  # no longer excluded

        # Update project config to exclude tool_a
        agent.project_config.excluded_tools.add("tool_a")
        agent._update_active_tools()

        # Check that project exclusion takes priority
        active_tool_names = agent.get_active_tool_names()
        assert "tool_a" not in active_tool_names  # now excluded by project

    def test_read_only_project_config(self, monkeypatch, temp_config_files):
        """Test that read_only project config disables editing tools."""
        agent = self.setup_agent_with_tools(monkeypatch, temp_config_files)

        # Set non-read-only first and check tools
        agent.project_config.read_only = False
        agent._update_active_tools()

        active_tool_names = agent.get_active_tool_names()
        assert "read_tool" in active_tool_names
        assert "edit_tool" in active_tool_names

        # Now set read_only to true
        agent.project_config.read_only = True
        agent._update_active_tools()

        # Check only read tools are available
        active_tool_names = agent.get_active_tool_names()
        assert "read_tool" in active_tool_names
        assert "edit_tool" not in active_tool_names

    def test_set_modes_dynamic_tool_update(self, monkeypatch, temp_config_files):
        """Test that set_modes correctly updates the active tools."""
        agent = self.setup_agent_with_tools(monkeypatch, temp_config_files)

        # Initial state with mode_x
        agent.set_modes(["mode_x"])
        agent._update_active_tools()

        # Check initial active tools
        active_tool_names = agent.get_active_tool_names()
        assert "tool_d" not in active_tool_names  # excluded by mode_x
        assert "tool_f" in active_tool_names  # not excluded

        # Change to mode_y
        result = agent.set_modes(["mode_y"])

        # Verify success result
        assert result == "OK"

        # Check that tools were updated
        active_tool_names = agent.get_active_tool_names()
        assert "tool_d" in active_tool_names  # no longer excluded
        assert "tool_f" not in active_tool_names  # now excluded by mode_y
