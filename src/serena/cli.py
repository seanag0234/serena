import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import click
from sensai.util import logging
from tqdm import tqdm

from serena.agent import SerenaAgent
from serena.config.context_mode import SerenaAgentContext, SerenaAgentMode
from serena.config.serena_config import ProjectConfig, SerenaConfig
from serena.constants import (
    DEFAULT_CONTEXT,
    DEFAULT_MODES,
    SERENA_MANAGED_DIR_IN_HOME,
    SERENAS_OWN_CONTEXT_YAMLS_DIR,
    SERENAS_OWN_MODE_YAMLS_DIR,
    USER_CONTEXT_YAMLS_DIR,
    USER_MODE_YAMLS_DIR,
)
from serena.mcp import SerenaMCPFactorySingleProcess
from serena.project import Project
from solidlsp.ls_config import Language

log = logging.getLogger(__name__)

# --------------------- Utilities -------------------------------------


def _open_in_editor(path: str) -> None:
    """Open the given file in the system's default editor or viewer."""
    editor = os.environ.get("EDITOR")
    try:
        if editor:
            subprocess.run([editor, path], check=False)
        elif sys.platform.startswith("win"):
            try:
                os.startfile(path)
            except OSError:
                subprocess.run(["notepad.exe", path], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"Failed to open {path}: {e}")


class ProjectType(click.ParamType):
    """ParamType allowing either a project name or a path to a project directory."""

    name = "[PROJECT_NAME|PROJECT_PATH]"

    def convert(self, value: str, param: Any, ctx: Any) -> str:
        path = Path(value).resolve()
        if path.exists() and path.is_dir():
            return str(path)
        return value


PROJECT_TYPE = ProjectType()


class AutoRegisteringGroup(click.Group):
    """
    A click.Group subclass that automatically registers any click.Command
    attributes defined on the class into the group.

    After initialization, it inspects its own class for attributes that are
    instances of click.Command (typically created via @click.command) and
    calls self.add_command(cmd) on each. This lets you define your commands
    as static methods on the subclass for IDE-friendly organization without
    manual registration.
    """

    def __init__(self, name: str, help: str):
        super().__init__(name=name, help=help)
        # Scan class attributes for click.Command instances and register them.
        for attr in dir(self.__class__):
            cmd = getattr(self.__class__, attr)
            if isinstance(cmd, click.Command):
                self.add_command(cmd)


class TopLevelCommands(AutoRegisteringGroup):
    """Root CLI group containing the core Serena commands."""

    def __init__(self) -> None:
        super().__init__(name="serena", help="Serena CLI commands. You can run `<command> --help` for more info on each command.")

    @staticmethod
    @click.command("start-mcp-server", help="Starts the Serena MCP server.")
    @click.option("--project", "project", type=PROJECT_TYPE, default=None, help="Path or name of project to activate at startup.")
    @click.option("--project-file", "project", type=PROJECT_TYPE, default=None, help="[DEPRECATED] Use --project instead.")
    @click.argument("project_file_arg", type=PROJECT_TYPE, required=False, default=None, metavar="")
    @click.option(
        "--context", type=str, default=DEFAULT_CONTEXT, show_default=True, help="Built-in context name or path to custom context YAML."
    )
    @click.option(
        "--mode",
        "modes",
        type=str,
        multiple=True,
        default=DEFAULT_MODES,
        show_default=True,
        help="Built-in mode names or paths to custom mode YAMLs.",
    )
    @click.option("--transport", type=click.Choice(["stdio", "sse"]), default="stdio", show_default=True, help="Transport protocol.")
    @click.option("--host", type=str, default="0.0.0.0", show_default=True)
    @click.option("--port", type=int, default=8000, show_default=True)
    @click.option("--enable-web-dashboard", type=bool, is_flag=False, default=None, help="Override dashboard setting in config.")
    @click.option("--enable-gui-log-window", type=bool, is_flag=False, default=None, help="Override GUI log window setting in config.")
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        default=None,
        help="Override log level in config.",
    )
    @click.option("--trace-lsp-communication", type=bool, is_flag=False, default=None, help="Whether to trace LSP communication.")
    @click.option("--tool-timeout", type=float, default=None, help="Override tool execution timeout in config.")
    def start_mcp_server(
        project: str | None,
        project_file_arg: str | None,
        context: str,
        modes: tuple[str, ...],
        transport: Literal["stdio", "sse"],
        host: str,
        port: int,
        enable_web_dashboard: bool | None,
        enable_gui_log_window: bool | None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None,
        trace_lsp_communication: bool | None,
        tool_timeout: float | None,
    ) -> None:
        project_file = project_file_arg or project
        factory = SerenaMCPFactorySingleProcess(context=context, project=project_file)
        server = factory.create_mcp_server(
            host=host,
            port=port,
            modes=modes,
            enable_web_dashboard=enable_web_dashboard,
            enable_gui_log_window=enable_gui_log_window,
            log_level=log_level,
            trace_lsp_communication=trace_lsp_communication,
            tool_timeout=tool_timeout,
        )
        if project_file_arg:
            log.warning(
                "Positional project arg is deprecated; use --project instead. Used: %s",
                project_file,
            )
        log.info("Starting MCP server …")
        server.run(transport=transport)

    @staticmethod
    @click.command("print-system-prompt", help="Print the system prompt for a project.")
    @click.argument("project", type=click.Path(exists=True), default=os.getcwd(), required=False)
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        default="WARNING",
        help="Log level for prompt generation.",
    )
    @click.option("--only-instructions", is_flag=True, help="Print only the initial instructions, without prefix/postfix.")
    @click.option(
        "--context", type=str, default=DEFAULT_CONTEXT, show_default=True, help="Built-in context name or path to custom context YAML."
    )
    @click.option(
        "--mode",
        "modes",
        type=str,
        multiple=True,
        default=DEFAULT_MODES,
        show_default=True,
        help="Built-in mode names or paths to custom mode YAMLs.",
    )
    def print_system_prompt(project: str, log_level: str, only_instructions: bool, context: str, modes: tuple[str, ...]) -> None:
        prefix = "You will receive access to Serena's symbolic tools. Below are instructions for using them, take them into account."
        postfix = "You begin by acknowledging that you understood the above instructions and are ready to receive tasks."
        from serena.tools.workflow_tools import InitialInstructionsTool

        lvl = logging.getLevelNamesMapping()[log_level.upper()]
        logging.configure(level=lvl)
        context_instance = SerenaAgentContext.load(context)
        mode_instances = [SerenaAgentMode.load(mode) for mode in modes]
        agent = SerenaAgent(
            project=os.path.abspath(project),
            serena_config=SerenaConfig(web_dashboard=False, log_level=lvl),
            context=context_instance,
            modes=mode_instances,
        )
        tool = agent.get_tool(InitialInstructionsTool)
        instr = tool.apply()
        if only_instructions:
            print(instr)
        else:
            print(f"{prefix}\n{instr}\n{postfix}")


class ModeCommands(AutoRegisteringGroup):
    """Group for 'mode' subcommands."""

    def __init__(self) -> None:
        super().__init__(name="mode", help="Manage Serena modes. You can run `mode <command> --help` for more info on each command.")

    @staticmethod
    @click.command("list", help="List available modes.")
    def list() -> None:
        mode_names = SerenaAgentMode.list_registered_mode_names()
        max_len_name = max(len(name) for name in mode_names) if mode_names else 20
        for name in mode_names:
            mode_yml_path = SerenaAgentMode.get_path(name)
            is_internal = Path(mode_yml_path).is_relative_to(SERENAS_OWN_MODE_YAMLS_DIR)
            descriptor = "(internal)" if is_internal else f"(at {mode_yml_path})"
            name_descr_string = f"{name:<{max_len_name + 4}}{descriptor}"
            click.echo(name_descr_string)

    @staticmethod
    @click.command("create", help="Create a new mode or copy an internal one.")
    @click.option(
        "--name",
        type=str,
        default=None,
        help="Name for the new mode. If --from-internal is passed may be left empty to create a mode of the same name, which will then override the internal mode.",
    )
    @click.option("--from-internal", "from_internal", type=str, default=None, help="Copy from an internal mode.")
    def create(name: str, from_internal: str) -> None:
        if not (name or from_internal):
            raise click.UsageError("Provide at least one of --name or --from-internal.")
        mode_name = name or from_internal
        dest = os.path.join(USER_MODE_YAMLS_DIR, f"{mode_name}.yml")
        src = (
            os.path.join(SERENAS_OWN_MODE_YAMLS_DIR, f"{from_internal}.yml")
            if from_internal
            else os.path.join(SERENAS_OWN_MODE_YAMLS_DIR, "mode.template.yml")
        )
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Internal mode '{from_internal}' not found in {SERENAS_OWN_MODE_YAMLS_DIR}. Available modes: {SerenaAgentMode.list_registered_mode_names()}"
            )
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        click.echo(f"Created mode '{mode_name}' at {dest}.")
        _open_in_editor(dest)

    @staticmethod
    @click.command("edit", help="Edit a custom mode YAML file.")
    @click.argument("mode_name")
    def edit(mode_name: str) -> None:
        path = os.path.join(USER_MODE_YAMLS_DIR, f"{mode_name}.yml")
        if not os.path.exists(path):
            if mode_name in SerenaAgentMode.list_registered_mode_names(include_user_modes=False):
                click.echo(
                    f"Mode '{mode_name}' is an internal mode and cannot be edited directly. "
                    f"Use 'mode create --from-internal {mode_name}' to create a custom mode that overrides it before editing."
                )
            else:
                click.echo(f"Custom mode '{mode_name}' not found. Create it with: mode create --name {mode_name}.")
            return
        _open_in_editor(path)

    @staticmethod
    @click.command("delete", help="Delete a custom mode file.")
    @click.argument("mode_name")
    def delete(mode_name: str) -> None:
        path = os.path.join(USER_MODE_YAMLS_DIR, f"{mode_name}.yml")
        if not os.path.exists(path):
            click.echo(f"Custom mode '{mode_name}' not found.")
            return
        os.remove(path)
        click.echo(f"Deleted custom mode '{mode_name}'.")


class ContextCommands(AutoRegisteringGroup):
    """Group for 'context' subcommands."""

    def __init__(self) -> None:
        super().__init__(
            name="context", help="Manage Serena contexts. You can run `context <command> --help` for more info on each command."
        )

    @staticmethod
    @click.command("list", help="List available contexts.")
    def list() -> None:
        context_names = SerenaAgentContext.list_registered_context_names()
        max_len_name = max(len(name) for name in context_names) if context_names else 20
        for name in context_names:
            context_yml_path = SerenaAgentContext.get_path(name)
            is_internal = Path(context_yml_path).is_relative_to(SERENAS_OWN_CONTEXT_YAMLS_DIR)
            descriptor = "(internal)" if is_internal else f"(at {context_yml_path})"
            name_descr_string = f"{name:<{max_len_name + 4}}{descriptor}"
            click.echo(name_descr_string)

    @staticmethod
    @click.command("create", help="Create a new context or copy an internal one.")
    @click.option(
        "--name",
        type=str,
        default=None,
        help="Name for the new context. If --from-internal is passed may be left empty to create a context of the same name, which will then override the internal context",
    )
    @click.option("--from-internal", "from_internal", type=str, default=None, help="Copy from an internal context.")
    def create(name: str, from_internal: str) -> None:
        if not (name or from_internal):
            raise click.UsageError("Provide at least one of --name or --from-internal.")
        ctx_name = name or from_internal
        dest = os.path.join(USER_CONTEXT_YAMLS_DIR, f"{ctx_name}.yml")
        src = (
            os.path.join(SERENAS_OWN_CONTEXT_YAMLS_DIR, f"{from_internal}.yml")
            if from_internal
            else os.path.join(SERENAS_OWN_CONTEXT_YAMLS_DIR, "context.template.yml")
        )
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Internal context '{from_internal}' not found in {SERENAS_OWN_CONTEXT_YAMLS_DIR}. Available contexts: {SerenaAgentContext.list_registered_context_names()}"
            )
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        click.echo(f"Created context '{ctx_name}' at {dest}.")
        _open_in_editor(dest)

    @staticmethod
    @click.command("edit", help="Edit a custom context YAML file.")
    @click.argument("context_name")
    def edit(context_name: str) -> None:
        path = os.path.join(USER_CONTEXT_YAMLS_DIR, f"{context_name}.yml")
        if not os.path.exists(path):
            if context_name in SerenaAgentContext.list_registered_context_names(include_user_contexts=False):
                click.echo(
                    f"Context '{context_name}' is an internal context and cannot be edited directly. "
                    f"Use 'context create --from-internal {context_name}' to create a custom context that overrides it before editing."
                )
            else:
                click.echo(f"Custom context '{context_name}' not found. Create it with: context create --name {context_name}.")
            return
        _open_in_editor(path)

    @staticmethod
    @click.command("delete", help="Delete a custom context file.")
    @click.argument("context_name")
    def delete(context_name: str) -> None:
        path = os.path.join(USER_CONTEXT_YAMLS_DIR, f"{context_name}.yml")
        if not os.path.exists(path):
            click.echo(f"Custom context '{context_name}' not found.")
            return
        os.remove(path)
        click.echo(f"Deleted custom context '{context_name}'.")


class SerenaConfigCommands(AutoRegisteringGroup):
    """Group for 'config' subcommands."""

    def __init__(self) -> None:
        super().__init__(name="config", help="Manage Serena configuration.")

    @staticmethod
    @click.command(
        "edit", help="Edit serena_config.yml in your default editor. Will create a config file from the template if no config is found."
    )
    def edit() -> None:
        config_path = os.path.join(SERENA_MANAGED_DIR_IN_HOME, "serena_config.yml")
        if not os.path.exists(config_path):
            SerenaConfig.generate_config_file(config_path)
        _open_in_editor(config_path)


class ProjectCommands(AutoRegisteringGroup):
    """Group for 'project' subcommands."""

    def __init__(self) -> None:
        super().__init__(
            name="project", help="Manage Serena projects. You can run `project <command> --help` for more info on each command."
        )

    @staticmethod
    @click.command("generate-yml", help="Generate a project.yml file.")
    @click.argument("project_path", type=click.Path(exists=True, file_okay=False), default=os.getcwd())
    @click.option("--language", type=str, default=None, help="Programming language; inferred if not specified.")
    def generate_yml(project_path: str, language: str | None = None) -> None:
        yml_path = os.path.join(project_path, ProjectConfig.rel_path_to_project_yml())
        if os.path.exists(yml_path):
            raise FileExistsError(f"Project file {yml_path} already exists.")
        lang_inst = None
        if language:
            try:
                lang_inst = Language[language.upper()]
            except KeyError:
                all_langs = [l.name.lower() for l in Language.iter_all(include_experimental=True)]
                raise ValueError(f"Unknown language '{language}'. Supported: {all_langs}")
        generated_conf = ProjectConfig.autogenerate(project_root=project_path, project_language=lang_inst)
        print(f"Generated project.yml with language {generated_conf.language.value} at {yml_path}.")

    @staticmethod
    @click.command("index", help="Index a project by saving symbols to the LSP cache.")
    @click.argument("project", type=click.Path(exists=True), default=os.getcwd(), required=False)
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        default="WARNING",
        help="Log level for indexing.",
    )
    def index(project: str, log_level: str = "WARNING") -> None:
        ProjectCommands._index_project(project, log_level)

    @staticmethod
    @click.command("index-deprecated", help="Deprecated alias for 'project index'.")
    @click.argument("project", type=click.Path(exists=True), default=os.getcwd(), required=False)
    @click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="WARNING")
    def index_deprecated(project: str, log_level: str = "WARNING") -> None:
        click.echo("Deprecated! Use `project index` instead.")
        ProjectCommands._index_project(project, log_level)

    @staticmethod
    def _index_project(project: str, log_level: str) -> None:
        lvl = logging.getLevelNamesMapping()[log_level.upper()]
        proj = Project.load(os.path.abspath(project))
        print(f"Indexing symbols in project {project}…")
        ls = proj.create_language_server(log_level=lvl)
        with ls.start_server():
            files = proj.gather_source_files()
            for i, f in enumerate(tqdm(files, desc="Indexing")):
                ls.request_document_symbols(f, include_body=False)
                ls.request_document_symbols(f, include_body=True)
                if (i + 1) % 10 == 0:
                    ls.save_cache()
            ls.save_cache()
        print(f"Symbols saved to {ls.cache_path}")


# Expose groups so we can reference them in pyproject.toml
mode = ModeCommands()
context = ContextCommands()
project = ProjectCommands()
config = SerenaConfigCommands()

# Expose toplevel commands for the same reason
top_level = TopLevelCommands()
start_mcp_server = top_level.start_mcp_server
index_project = project.index_deprecated

# needed for the help script to work - register all subcommands to the top-level group
for subgroup in (mode, context, project, config):
    top_level.add_command(subgroup)


def get_help() -> str:
    """Retrieve the help text for the top-level Serena CLI."""
    return top_level.get_help(click.Context(top_level, info_name="serena"))
