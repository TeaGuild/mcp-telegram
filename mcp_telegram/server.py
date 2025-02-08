#!/usr/bin/env python3
"""
MCP Telegram Bridge Server

This module provides the MCP server interface for the Telegram bridge.
"""
import os
import sys
import json
import logging
import asyncio
from typing import Any, Dict, List, Literal, TypeVar, Callable
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
import anyio
from mcp.server.stdio import stdio_server
from pyrogram import Client
from pyrogram.types import Message
from pyrogram.errors import (
    UsernameNotOccupied,
    PeerIdInvalid,
    ChannelInvalid,
    ChatAdminRequired,
    UserNotParticipant,
    FloodWait,
    SlowmodeWait,
    ChatWriteForbidden,
    ChannelPrivate,
    UserDeactivated,
    UserDeactivatedBan
)

import mcp.types as types
from mcp.server import FastMCP
from mcp.server.fastmcp import Context

# Force anyio to use asyncio backend
os.environ["ANYIO_BACKEND"] = "asyncio"

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Add loggers for key components
pyrogram_logger = logging.getLogger('pyrogram')
pyrogram_logger.setLevel(logging.INFO)
mcp_logger = logging.getLogger('mcp')
mcp_logger.setLevel(logging.DEBUG)
asyncio_logger = logging.getLogger('asyncio')
asyncio_logger.setLevel(logging.DEBUG)

T = TypeVar('T')

async def handle_flood_wait(e: FloodWait, action: str):
    """Common handler for FloodWait errors."""
    logger.warning(f"Hit FloodWait in {action}, waiting {e.value} seconds")
    await asyncio.sleep(e.value)

async def retry_with_flood_handling(func: Callable[..., T], action: str, *args, **kwargs) -> T:
    """Generic retry handler for functions that may hit FloodWait."""
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except FloodWait as e:
            if attempt == max_retries - 1:
                raise
            
            delay = e.value + (base_delay * (2 ** attempt))
            logger.warning(f"Hit FloodWait in {action}, waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(delay)
    
    return await func(*args, **kwargs)  # Final attempt

async def get_chat_history_with_retry(
    chat_id: int | str,
    limit: int = 0,
    offset: int = 0,
    offset_id: int | str | None = None,
    **kwargs
):
    """Wrapper for get_chat_history with retry on flood."""
    messages = []
    logger.debug(f"get_chat_history_with_retry called with chat_id={chat_id}, limit={limit}, offset={offset}, offset_id={offset_id}, kwargs={kwargs}")
    parsed_offset_id = parse_message_id(offset_id) or 0
    
    async def fetch_messages():
        async for msg in mcp.client.get_chat_history(
            chat_id=chat_id,
            limit=limit,
            offset=offset,
            offset_id=parsed_offset_id,
            **kwargs
        ):
            try:
                logger.debug(f"Got message: id={msg.id}, type={type(msg)}")
                messages.append(msg)
                if len(messages) >= limit:
                    logger.debug("Reached message limit, breaking")
                    break
            except FloodWait as e:
                await handle_flood_wait(e, "get_chat_history")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise
    
    try:
        await retry_with_flood_handling(fetch_messages, "get_chat_history")
    except Exception as e:
        logger.error(f"Error in get_chat_history: {e}, chat_id={chat_id}, limit={limit}, offset={offset}, offset_id={offset_id}, kwargs={kwargs}")
        raise
    
    logger.debug(f"Returning {len(messages)} messages")
    return messages

async def search_messages_with_retry(chat_id: int | str, query: str, **kwargs):
    """Wrapper for search_messages with retry on flood."""
    messages = []
    logger.debug(f"search_messages_with_retry called with chat_id={chat_id}, query={query}, kwargs={kwargs}")
    
    async def fetch_messages():
        async for msg in mcp.client.search_messages(chat_id=chat_id, query=query, **kwargs):
            try:
                logger.debug(f"Got message: id={msg.id}, type={type(msg)}")
                messages.append(msg)
                if len(messages) >= kwargs.get('limit', 100):
                    logger.debug("Reached message limit, breaking")
                    break
            except FloodWait as e:
                await handle_flood_wait(e, "search_messages")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise
    
    try:
        await retry_with_flood_handling(fetch_messages, "search_messages")
    except Exception as e:
        logger.error(f"Error in search_messages: {e}, chat_id={chat_id}, query={query}, kwargs={kwargs}")
        raise
    
    logger.debug(f"Returning {len(messages)} messages")
    return messages

def get_channel_id(id: int) -> int:
    """Convert negative channel ID to its positive counterpart."""
    return -1000000000000 - id if id < 0 else id

def get_input_peer(chat_id: int, access_hash: int | None = None, peer_type: str = "user") -> Any:
    """Convert chat ID and access hash into appropriate InputPeer type.
    
    Args:
        chat_id: The chat ID
        access_hash: The access hash (required for users and channels)
        peer_type: One of "user", "chat", "channel"
    """
    if peer_type == "user":
        from pyrogram.raw.types import InputPeerUser
        return InputPeerUser(user_id=chat_id, access_hash=access_hash or 0)
    elif peer_type == "chat":
        from pyrogram.raw.types import InputPeerChat
        return InputPeerChat(chat_id=-chat_id if chat_id < 0 else chat_id)
    elif peer_type == "channel":
        from pyrogram.raw.types import InputPeerChannel
        return InputPeerChannel(
            channel_id=get_channel_id(chat_id) if chat_id < 0 else chat_id,
            access_hash=access_hash or 0
        )
    else:
        from pyrogram.raw.types import InputPeerEmpty
        return InputPeerEmpty()

def parse_message_id(message_id: str | int | None) -> int | None:
    """Parse message ID into the appropriate type.
    
    Handles:
    - Integer IDs (as strings or actual integers)
    - None values
    """
    if message_id is None:
        return None
        
    try:
        return int(str(message_id))
    except (ValueError, AttributeError):
        return None

def parse_chat_id(chat: str) -> str | int:
    """Parse chat identifier into the appropriate type.
    
    Handles:
    - Usernames (string starting with @ or not)
    - Integer IDs (as strings or actual integers)
    - Phone numbers (strings)
    - Channel IDs (converts to proper format)
    """
    chat = str(chat).lstrip('@')
    try:
        # Try to convert to integer if it looks like an ID
        if chat.lstrip('-').isdigit():
            chat_id = int(chat)
            # Handle channel IDs
            if chat_id < 0:
                return get_channel_id(chat_id)
            return chat_id
        return chat
    except (ValueError, AttributeError):
        return chat

def serialize_message(message: Message) -> Dict[str, Any]:
    """Convert a Telegram message to a clean JSON structure"""
    data = {
        "id": message.id,
        "date": message.date.isoformat() if message.date else None,
        "edit_date": message.edit_date.isoformat() if message.edit_date else None,
        "text": message.text or message.caption,
        "from_user": {
            "id": message.from_user.id if message.from_user else None,
            "name": message.from_user.first_name if message.from_user else None,
            "username": message.from_user.username if message.from_user else None
        } if message.from_user else None,
        "chat": {
            "id": message.chat.id,
            "title": message.chat.title,
            "username": message.chat.username
        } if message.chat else None,
        "reply_to_message_id": parse_message_id(message.reply_to_message_id),
        "forward_from": message.forward_from.id if message.forward_from else None,
        "views": getattr(message, 'views', None),
        "media": None
    }
    
    # Add media information if present
    if message.media:
        if message.document:
            data["media"] = {
                "type": "document",
                "mime_type": message.document.mime_type,
                "file_size": message.document.file_size,
                "filename": message.document.file_name
            }
        elif message.photo:
            # Photos come as a list of different sizes, get the largest one
            photo = message.photo[-1]
            data["media"] = {
                "type": "photo",
                "file_id": photo.file_id,
                "file_unique_id": photo.file_unique_id,
                "width": photo.width,
                "height": photo.height,
                "file_size": photo.file_size,
                "sizes": [{
                    "file_id": size.file_id,
                    "file_unique_id": size.file_unique_id,
                    "width": size.width,
                    "height": size.height,
                    "file_size": size.file_size
                } for size in message.photo]
            }
        else:
            data["media"] = {"type": str(message.media)}
    
    return data

class TelegramBridge(FastMCP):
    def __init__(self):
        super().__init__("telegram-bridge")
        
        # Load environment variables
        load_dotenv()

        # Validate credentials
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.session_string = os.getenv("TELEGRAM_SESSION_STRING")

        if not all([self.api_id, self.api_hash, self.session_string]):
            raise ValueError(
                "Missing required environment variables. Please set TELEGRAM_API_ID, "
                "TELEGRAM_API_HASH, and TELEGRAM_SESSION_STRING"
            )

        # Initialize Telegram client
        self.client = Client(
            "mcp_telegram",
            api_id=self.api_id,
            api_hash=self.api_hash,
            session_string=self.session_string,
            in_memory=True,
            no_updates=True  # Disable updates since we don't need them
        )

    async def _run_with_transport(self, transport: Literal["stdio", "sse"]):
        """Internal method to run with specific transport"""
        # Start client
        logger.info("Starting Telegram client...")
        await self.client.start()
        logger.info("Telegram client started successfully")

        try:
            if transport == "stdio":
                async with stdio_server() as (read_stream, write_stream):
                    await self._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp_server.create_initialization_options(),
                    )
            else:  # transport == "sse"
                await super().run_sse_async()
        finally:
            if self.client.is_connected:
                logger.info("Stopping Telegram client...")
                await self.client.stop()
                logger.info("Telegram client stopped successfully")

    def run(self, transport: Literal["stdio", "sse"] = "stdio") -> None:
        """Run the FastMCP server with Telegram client."""
        # Use anyio's run() to start the event loop
        anyio.run(lambda: self._run_with_transport(transport))

# Create the MCP server that the CLI will look for
mcp = TelegramBridge()

async def handle_chat_errors(chat: str, ctx: Context | None, action: str):
    """Common error handling for chat operations.
    
    Args:
        chat: Chat identifier
        ctx: Context for error reporting
        action: Description of the action being performed (for logging)
    
    Returns:
        The resolved chat object
        
    Raises:
        Various Telegram exceptions with appropriate error messages
    """
    try:
        logger.debug(f"Resolving chat info for: {chat}")
        chat_obj = await retry_with_flood_handling(mcp.client.get_chat, "get_chat", chat)
        logger.debug(f"Chat resolved: id={chat_obj.id}, type={chat_obj.type}")
        return chat_obj
    except (UsernameNotOccupied, PeerIdInvalid, ChannelInvalid) as e:
        if ctx:
            ctx.error(f"Chat '{chat}' not found. Please check if the username/ID is correct.")
        logger.error(f"Chat not found while {action}: {chat} - {e}")
        raise
    except (ChatAdminRequired, UserNotParticipant, ChannelPrivate) as e:
        if ctx:
            ctx.error(f"No access to chat '{chat}'. You need to join the chat first.")
        logger.error(f"No access to chat while {action}: {chat} - {e}")
        raise
    except (FloodWait, SlowmodeWait) as e:
        if ctx:
            ctx.error(f"Rate limited. Please wait {e.value} seconds before trying again.")
        logger.error(f"Rate limit hit while {action}: {chat} - {e}")
        raise
    except (UserDeactivated, UserDeactivatedBan) as e:
        if ctx:
            ctx.error("Your account has been deactivated or banned.")
        logger.error(f"Account deactivated/banned while {action}: {e}")
        raise

@mcp.tool()
async def get_messages(
    chat: str,
    limit: int = 10,
    offset_id: str | int | None = None,
    from_date: str | None = None,
    has_media: bool | None = None,
    ctx: Context | None = None
) -> str:
    """Get messages from a chat with optional filters.
    
    Args:
        chat: Username, phone, or chat ID
        limit: Maximum number of messages to return
        offset_id: Get messages before this ID
        from_date: Get messages from this date (ISO format)
        has_media: Filter messages with media
    """
    messages = []
    if not chat:
        raise ValueError("chat parameter is required")
    chat = parse_chat_id(chat)
    logger.debug(f"Getting messages from chat: {chat} (parsed from input)")
    
    try:
        chat_obj = await handle_chat_errors(chat, ctx, "getting messages")
        
        date_filter = None
        if from_date:
            date_filter = datetime.fromisoformat(from_date)

        # Get messages with retry
        parsed_offset = parse_message_id(offset_id)
        logger.debug(f"Getting chat history with: id={chat_obj.id}, limit={limit}, offset={parsed_offset}")
        
        messages_list = await get_chat_history_with_retry(
            chat_id=chat_obj.id,
            limit=limit,
            offset=0,
            offset_id=parsed_offset
        )
        logger.debug(f"Retrieved {len(messages_list)} messages from history")
        
        for message in messages_list:
            logger.debug(f"Processing message: id={message.id}, date={message.date}")
            if date_filter and message.date < date_filter:
                continue
            if has_media and not message.media:
                continue
            
            try:
                serialized = serialize_message(message)
                messages.append(serialized)
                logger.debug(f"Serialized message {message.id} successfully")
                if ctx:
                    await ctx.report_progress(len(messages), limit)
                    ctx.info(f"Retrieved message {len(messages)}/{limit}")
            except Exception as e:
                logger.error(f"Error serializing message {message.id}: {e}")
                continue

        logger.debug(f"Successfully processed {len(messages)} messages")
        result = json.dumps(messages, indent=2)
        logger.debug("Messages serialized to JSON successfully")
        return result
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting messages from chat {chat}: {e}")
        logger.error(f"Error getting messages from chat {chat}: {e}")
        raise

@mcp.tool()
async def search_messages(
    query: str,
    chats: List[str] | None = None,
    limit: int = 10,
    ctx: Context | None = None
) -> str:
    """Search for messages across chats.
    
    Args:
        query: Search query
        chats: List of chats to search in
        limit: Maximum number of results
    """
    results = []
    chats = chats or ["me"]

    for chat in chats:
        chat = parse_chat_id(chat)
        try:
            try:
                chat_obj = await handle_chat_errors(chat, ctx, "searching messages")
            except (UsernameNotOccupied, PeerIdInvalid, ChannelInvalid,
                    ChatAdminRequired, UserNotParticipant, ChannelPrivate,
                    FloodWait, SlowmodeWait) as e:
                # For search, we want to continue to next chat on most errors
                logger.warning(f"Skipping chat {chat} due to error: {e}")
                continue
            except (UserDeactivated, UserDeactivatedBan) as e:
                # These are fatal errors that should stop the entire operation
                raise
            
            # Search messages with retry
            messages_list = await search_messages_with_retry(
                chat_id=chat_obj.id,
                query=query,
                limit=limit
            )
            
            for message in messages_list:
                results.append(serialize_message(message))
                if len(results) >= limit:
                    break
                if ctx:
                    await ctx.report_progress(len(results), limit)
                    ctx.info(f"Found message in {chat}, total: {len(results)}")
        except Exception as e:
            if ctx:
                ctx.error(f"Error searching in chat {chat}: {e}")
            logger.error(f"Error searching in chat {chat}: {e}")
            continue

    return json.dumps(results, indent=2)

@mcp.tool()
async def get_chat_info(
    chat: str,
    ctx: Context | None = None
) -> str:
    """Get detailed information about a chat.
    
    Args:
        chat: Username, phone, or chat ID
    """
    chat = parse_chat_id(chat)
    
    try:
        chat_obj = await handle_chat_errors(chat, ctx, "getting chat info")

        if str(chat_obj.type) in ['supergroup', 'channel']:
            info = {
                "id": chat_obj.id,
                "title": chat_obj.title,
                "type": str(chat_obj.type),
                "username": chat_obj.username,
                "description": chat_obj.description,
                "member_count": chat_obj.members_count if hasattr(chat_obj, 'members_count') else None,
                "is_verified": chat_obj.is_verified if hasattr(chat_obj, 'is_verified') else None
            }
        else:
            info = {
                "id": chat_obj.id,
                "type": str(chat_obj.type),
                "first_name": chat_obj.first_name if hasattr(chat_obj, 'first_name') else None,
                "last_name": chat_obj.last_name if hasattr(chat_obj, 'last_name') else None,
                "username": chat_obj.username if hasattr(chat_obj, 'username') else None
            }

        return json.dumps(info, indent=2)
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting info for chat {chat}: {e}")
        logger.error(f"Error getting info for chat {chat}: {e}")
        raise

if __name__ == "__main__":
    try:
        # Run the MCP server
        logger.info("Starting MCP server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
