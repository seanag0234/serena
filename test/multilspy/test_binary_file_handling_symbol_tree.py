"""
Test for handling binary/unreadable files in request_full_symbol_tree.
"""

from multilspy.language_server import SyncLanguageServer
from multilspy.lsp_protocol_handler.lsp_types import SymbolKind


def test_full_symbol_tree_with_binary_files(language_server: SyncLanguageServer):
    """
    Test that request_full_symbol_tree properly handles unreadable files.

    It should include readable files in the result but skip unreadable ones.
    """
    # Get the symbol tree for the binary_test directory
    binary_test_path = "binary_test"
    symbol_tree = language_server.request_full_symbol_tree(within_relative_path=binary_test_path)

    # Verify that we got a result
    assert len(symbol_tree) == 1

    # Verify the package symbol is correct
    package_symbol = symbol_tree[0]
    assert package_symbol["name"] == "binary_test"
    assert package_symbol["kind"] == SymbolKind.Package
    assert "children" in package_symbol

    # Get the names of the file symbols (children of the package)
    file_symbols = package_symbol["children"]
    file_names = [file_symbol["name"] for file_symbol in file_symbols]

    # Verify that the readable file is included in the result
    assert "readable" in file_names

    # Verify that all file symbols have the correct kind
    assert all(file_symbol["kind"] == SymbolKind.File for file_symbol in file_symbols)

    # Find the readable file symbol
    readable_symbol = next((s for s in file_symbols if s["name"] == "readable"), None)
    assert readable_symbol is not None

    # Verify that the readable file has children (symbols defined in the file)
    assert "children" in readable_symbol
    assert len(readable_symbol["children"]) > 0

    # Verify the symbols in the readable file
    symbol_names = [symbol["name"] for symbol in readable_symbol["children"]]
    assert "hello" in symbol_names
    assert "TestClass" in symbol_names
    assert "test_variable" in symbol_names

    # Verify that the unreadable file is not included
    assert "unreadable" not in file_names
