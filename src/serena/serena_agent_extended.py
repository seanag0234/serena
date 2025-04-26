"""
Modified SerenaAgent class to support context and mode configurations.
"""
import os
import platform
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import Union

from multilspy import SyncLanguageServer
from serena import serena_version
from serena.agent import (
    LOG_FORMAT,
    ActivateProjectTool,
    GetActiveProjectTool,
    LinesRead,
    MemoriesManager,
    ProjectConfig,
    SerenaConfig,
    SymbolManager,
    Tool,
    iter_tool_classes,
    log,
)
from serena.agent import SerenaAgent as OriginalSerenaAgent
from serena.context_mode import ContextConfig, ModeConfig, load_context_config, load_mode_config, resolve_tool_activations
from serena.llm.prompt_factory import PromptFactory


class SerenaAgent(OriginalSerenaAgent):
    def __init__(
        self, 
        project_file_path: str | None = None, 
        project_activation_callback: Callable[[], None] | None = None,
        context_name_or_path: str | None = None,
        mode_names_or_paths: list[str] | None = None
    ):
        """
        :param project_file_path: the configuration file (.yml) of the project to load immediately;
            if None, do not load any project (must use project selection tool to activate a project).
            If a project is provided, the corresponding language server will be started.
        :param project_activation_callback: a callback function to be called when a project is activated.
        :param context_name_or_path: Name or path to a context configuration file
        :param mode_names_or_paths: List of names or paths to mode configuration files
        """
        # obtain serena configuration
        self.serena_config = SerenaConfig()

        # open GUI log window if enabled
        self._gui_log_handler: Union["GuiLogViewerHandler", None] = None  # noqa
        if self.serena_config.gui_log_window_enabled:
            if platform.system() == "Darwin":
                log.warning("GUI log window is not supported on macOS")
            else:
                # even importing on macOS may fail if tkinter dependencies are unavailable (depends on Python interpreter installation
                # which uv used as a base, unfortunately)
                from serena.gui_log_viewer import GuiLogViewer, GuiLogViewerHandler

                log_level = self.serena_config.gui_log_window_level
                if Logger.root.level > log_level:
                    log.info(f"Root logger level is higher than GUI log level; changing the root logger level to {log_level}")
                    Logger.root.setLevel(log_level)
                self._gui_log_handler = GuiLogViewerHandler(GuiLogViewer(title="Serena Logs"), level=log_level, format_string=LOG_FORMAT)
                Logger.root.addHandler(self._gui_log_handler)

        log.info(f"Starting Serena server (version={serena_version()}, process id={os.getpid()}, parent process id={os.getppid()})")
        log.info("Available projects: {}".format(", ".join(self.serena_config.project_names)))

        self.prompt_factory = PromptFactory()
        self._project_activation_callback = project_activation_callback

        # Initialize context and modes
        self.context_config: ContextConfig | None = None
        self.mode_configs: list[ModeConfig] = []
        
        # Load context and modes if provided
        if context_name_or_path:
            self.context_config = load_context_config(context_name_or_path)
            log.info(f"Loaded context: {self.context_config.name}")
        
        if mode_names_or_paths:
            for mode_name_or_path in mode_names_or_paths:
                mode_config = load_mode_config(mode_name_or_path)
                self.mode_configs.append(mode_config)
            log.info(f"Loaded modes: {', '.join([mode.name for mode in self.mode_configs])}")
        
        # project-specific instances, which will be initialized upon project activation
        self.project_config: ProjectConfig | None = None
        self.language_server: SyncLanguageServer | None = None
        self.symbol_manager: SymbolManager | None = None
        self.memories_manager: MemoriesManager | None = None
        self.lines_read: LinesRead | None = None

        # find all tool classes and instantiate them
        self._all_tools: dict[type[Tool], Tool] = {}
        for tool_class in iter_tool_classes():
            tool_instance = tool_class(self)
            if not self.serena_config.enable_project_activation:
                if tool_class in (GetActiveProjectTool, ActivateProjectTool):
                    log.info(f"Excluding tool '{tool_instance.get_name()}' because project activation is disabled in configuration")
                    continue
            self._all_tools[tool_class] = tool_instance
        
        # Apply context and mode tool settings
        self._apply_context_and_mode_tool_settings()
        
        # If GUI log window is enabled, set the tool names for highlighting
        if self._gui_log_handler is not None:
            tool_names = [tool.get_name() for tool in self._active_tools.values()]
            self._gui_log_handler.log_viewer.set_tool_names(tool_names)

        # activate a project configuration (if provided or if there is only a single project available)
        project_config: ProjectConfig | None = None
        if project_file_path is not None:
            if not os.path.exists(project_file_path):
                raise FileNotFoundError(f"Project file not found: {project_file_path}")
            log.info(f"Loading project configuration from {project_file_path}")
            project_config = ProjectConfig.from_yml(Path(project_file_path))
        else:
            if len(self.serena_config.projects) == 1:
                project_config = self.serena_config.get_project_configuration(self.serena_config.project_names[0])
            elif len(self.serena_config.projects) == 0:
                raise RuntimeError(f"No projects found in {SerenaConfig.CONFIG_FILE} and no project file specified.")
            
        if project_config is not None:
            self.activate_project(project_config)
        else:
            if not self.serena_config.enable_project_activation:
                raise ValueError("Tool-based project activation is disabled in the configuration but no project file was provided.")
    
    def _apply_context_and_mode_tool_settings(self) -> None:
        """Apply tool settings from context and mode configurations."""
        # Start with all tools
        self._active_tools = dict(self._all_tools)
        
        # Apply context and mode tool settings
        if self.context_config or self.mode_configs:
            included_tools, excluded_tools = resolve_tool_activations(self.context_config, self.mode_configs)
            
            # Apply explicit includes if any are specified
            if included_tools:
                self._active_tools = {
                    key: tool for key, tool in self._all_tools.items() 
                    if tool.get_name() in included_tools
                }
            
            # Apply excludes
            if excluded_tools:
                self._active_tools = {
                    key: tool for key, tool in self._active_tools.items() 
                    if tool.get_name() not in excluded_tools
                }
                
            log.info(f"Active tools after context/mode configuration ({len(self._active_tools)}): {', '.join(self.get_active_tool_names())}")
    
    def set_modes(self, mode_names_or_paths: list[str]) -> None:
        """
        Set new modes for the agent.
        
        :param mode_names_or_paths: List of mode names or paths
        """
        # Clear current modes
        self.mode_configs = []
        
        # Load new modes
        for mode_name_or_path in mode_names_or_paths:
            mode_config = load_mode_config(mode_name_or_path)
            self.mode_configs.append(mode_config)
        
        log.info(f"Updated modes: {', '.join([mode.name for mode in self.mode_configs])}")
        
        # Reapply tool settings
        self._apply_context_and_mode_tool_settings()
        
        # If project is active, apply project tool exclusions
        if self.project_config:
            self._apply_project_tool_exclusions()
    
    def _apply_project_tool_exclusions(self) -> None:
        """Apply tool exclusions from the project configuration."""
        if not self.project_config:
            return
        
        # Apply project-specific exclusions
        if self.project_config.excluded_tools:
            self._active_tools = {
                key: tool for key, tool in self._active_tools.items() 
                if tool.get_name() not in self.project_config.excluded_tools
            }
            log.info(f"Active tools after project exclusions ({len(self._active_tools)}): {', '.join(self.get_active_tool_names())}")
        
        # If read_only mode is enabled, exclude all editing tools
        if self.project_config.read_only:
            self._active_tools = {key: tool for key, tool in self._active_tools.items() if not key.can_edit()}
            log.info(
                f"Project is in read-only mode. Editing tools excluded. Active tools ({len(self._active_tools)}): "
                f"{', '.join(self.get_active_tool_names())}"
            )
    
    def activate_project(self, project_config: ProjectConfig) -> None:
        """
        Activate a project configuration.
        
        :param project_config: The project configuration to activate
        """
        log.info(f"Activating {project_config}")
        self.project_config = project_config

        # Apply project tool exclusions
        self._apply_project_tool_exclusions()

        # start the language server
        self.reset_language_server()
        assert self.language_server is not None

        # initialize project-specific instances
        self.symbol_manager = SymbolManager(self.language_server, self)
        self.memories_manager = MemoriesManager(os.path.join(self.project_config.get_serena_managed_dir(), "memories"))
        self.lines_read = LinesRead()

        if self._project_activation_callback is not None:
            self._project_activation_callback()
            
    def get_system_prompt(self) -> str:
        """
        Get the system prompt with context and mode information.
        
        :return: The system prompt
        """
        return self.prompt_factory.create_system_prompt(
            context=self.context_config,
            modes=self.mode_configs
        )
