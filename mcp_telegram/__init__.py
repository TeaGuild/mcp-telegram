"""
MCP Telegram Bridge

A Model Context Protocol server that provides structured access to Telegram data.
This MCP acts as a data bridge, allowing AI assistants to efficiently access and
analyze Telegram messages, media, and chat information.

Features:
- Access message history from any chat
- Search for specific messages
- Get detailed chat information
- Send messages (with appropriate disclaimers)
- Access and analyze media content
"""

from .server import mcp

__version__ = "0.1.0"
__author__ = "MCP Telegram Contributors"

__all__ = ["mcp"]
