"""
Tests for the context and mode configuration feature.
"""
from unittest import mock

import pytest

from serena.context_mode import (
    ContextConfig,
    ModeConfig,
    PromptAdjustments,
    ToolSettings,
    load_context_config,
    load_mode_config,
    resolve_tool_activations,
)
from serena.serena_agent_extended import SerenaAgent


# Mock configurations for testing
@pytest.fixture
def mock_context_config():
    return ContextConfig(
        name="test-context",
        description="Test Context",
        tool_settings=ToolSettings(
            included_tools=["tool1", "tool2"],
            excluded_tools=["tool3"]
        ),
        prompt_adjustments=PromptAdjustments(
            description="Context Description",
            instructions=["Context Instruction 1", "Context Instruction 2"]
        )
    )


@pytest.fixture
def mock_mode_configs():
    return [
        ModeConfig(
            name="test-mode-1",
            description="Test Mode 1",
            tool_settings=ToolSettings(
                included_tools=["tool4"],
                excluded_tools=["tool5"]
            ),
            prompt_adjustments=PromptAdjustments(
                description="Mode 1 Description",
                instructions=["Mode 1 Instruction"]
            )
        ),
        ModeConfig(
            name="test-mode-2",
            description="Test Mode 2",
            tool_settings=ToolSettings(
                included_tools=["tool6"],
                excluded_tools=["tool7"]
            ),
            prompt_adjustments=PromptAdjustments(
                description="Mode 2 Description",
                instructions=["Mode 2 Instruction"],
                examples=["Mode 2 Example"]
            )
        )
    ]


def test_resolve_tool_activations(mock_context_config, mock_mode_configs):
    """Test that tool activations are correctly resolved."""
    included, excluded = resolve_tool_activations(mock_context_config, mock_mode_configs)
    
    # Context takes precedence
    assert "tool1" in included
    assert "tool2" in included
    assert "tool3" in excluded
    
    # Modes are also applied
    assert "tool4" in included
    assert "tool5" in excluded
    assert "tool6" in included
    assert "tool7" in excluded


def test_resolve_tool_activations_conflict_detection():
    """Test that conflicts between modes are detected."""
    mode1 = ModeConfig(
        name="mode1",
        description="Mode 1",
        tool_settings=ToolSettings(
            included_tools=["tool1"],
            excluded_tools=[]
        ),
        prompt_adjustments=PromptAdjustments(
            description="",
            instructions=[]
        )
    )
    
    mode2 = ModeConfig(
        name="mode2",
        description="Mode 2",
        tool_settings=ToolSettings(
            included_tools=[],
            excluded_tools=["tool1"]
        ),
        prompt_adjustments=PromptAdjustments(
            description="",
            instructions=[]
        )
    )
    
    # This should raise a conflict error
    with pytest.raises(ValueError):
        resolve_tool_activations(None, [mode1, mode2])


def test_context_priority_over_modes(mock_context_config):
    """Test that context takes precedence over modes."""
    # Mode tries to exclude tool1, but context includes it
    mode = ModeConfig(
        name="mode",
        description="Mode",
        tool_settings=ToolSettings(
            included_tools=[],
            excluded_tools=["tool1", "tool2"]
        ),
        prompt_adjustments=PromptAdjustments(
            description="",
            instructions=[]
        )
    )
    
    included, excluded = resolve_tool_activations(mock_context_config, [mode])
    
    # Context includes should override mode excludes
    assert "tool1" in included
    assert "tool2" in included
    assert "tool3" in excluded



def test_load_context_and_modes():
    """Test that context and mode configurations can be loaded correctly."""
    with mock.patch('serena.context_mode.resolve_path_or_name') as mock_resolve_path:
        mock_path = mock.MagicMock()
        mock_path.exists.return_value = True
        mock_resolve_path.return_value = mock_path

        with mock.patch('serena.context_mode.ContextConfig.from_yml') as mock_context_from_yml:
            with mock.patch('serena.context_mode.ModeConfig.from_yml') as mock_mode_from_yml:
                # Setup mocks
                mock_context = mock.MagicMock()
                mock_context.name = "test-context"
                mock_mode = mock.MagicMock()
                mock_mode.name = "test-mode"

                mock_context_from_yml.return_value = mock_context
                mock_mode_from_yml.return_value = mock_mode

                # Test loading context
                context = load_context_config("test-context")
                mock_resolve_path.assert_called_with("test-context", "contexts")
                assert context == mock_context

                # Test loading mode
                mode = load_mode_config("test-mode")
                mock_resolve_path.assert_called_with("test-mode", "modes")
                assert mode == mock_mode

def test_set_modes():
    """Test that set_modes correctly updates modes."""
    # Create a partial mock of SerenaAgent
    with mock.patch('serena.serena_agent_extended.SerenaAgent.__init__', return_value=None):
        with mock.patch('serena.serena_agent_extended.load_mode_config') as mock_load_mode:
            # Setup mocks
            mode1 = mock.MagicMock()
            mode1.name = "test-mode-1"
            mode2 = mock.MagicMock()
            mode2.name = "test-mode-2"
            mode3 = mock.MagicMock()
            mode3.name = "test-mode-3"
            
            mock_load_mode.side_effect = [mode1, mode2, mode3]
            
            # Create agent
            agent = SerenaAgent()
            agent.context_config = mock.MagicMock()
            agent.mode_configs = []
            agent._apply_context_and_mode_tool_settings = mock.MagicMock()
            agent._apply_project_tool_exclusions = mock.MagicMock()
            agent.project_config = None
            
            # Call set_modes
            agent.set_modes(["test-mode-1", "test-mode-2"])
            
            # Verify that load_mode_config was called for each mode
            assert mock_load_mode.call_count == 2
            mock_load_mode.assert_any_call("test-mode-1")
            mock_load_mode.assert_any_call("test-mode-2")
            
            # Verify that mode configs were updated
            assert len(agent.mode_configs) == 2
            assert agent.mode_configs[0] == mode1
            assert agent.mode_configs[1] == mode2
            
            # Verify that tool settings were reapplied
            agent._apply_context_and_mode_tool_settings.assert_called_once()
            
            # Since project_config is None, _apply_project_tool_exclusions should not be called
            agent._apply_project_tool_exclusions.assert_not_called()
            
            # Reset mocks
            agent._apply_context_and_mode_tool_settings.reset_mock()
            
            # Now test with a project_config
            agent.project_config = mock.MagicMock()
            
            # Call set_modes
            agent.set_modes(["test-mode-3"])
            
            # Verify that mode configs were updated
            assert len(agent.mode_configs) == 1
            assert agent.mode_configs[0] == mode3
            
            # Verify that tool settings were reapplied
            agent._apply_context_and_mode_tool_settings.assert_called_once()
            
            # _apply_project_tool_exclusions should be called this time
            agent._apply_project_tool_exclusions.assert_called_once()
