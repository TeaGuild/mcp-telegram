#!/usr/bin/env python3
"""
MCP Telegram Bridge Server

This module provides the MCP server interface for the Telegram bridge.
"""
import io
import os
import sys
import json
import sqlite3
import logging
import asyncio
import pathlib
from xdg import xdg_data_home
from typing import Any, Dict, List, Literal, TypeVar, Callable
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
import anyio
from mcp.server.stdio import stdio_server
from PIL import Image as PILImage
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from pyrogram import Client, enums
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

async def resolve_chat(chat_ent: Any) -> str:
    logger.debug(f"IN: `{chat_ent}` ({type(chat_ent)})")

    chat_ent = str(chat_ent)

    return chat_ent
    
    # if it looks like a telegram __id__ (e.g. all (or with a `-`) numbers), convert it to int
    # else try to resolve via pyrogram, get the entity

    


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
    """Wrapper for get_chat_history with retry on flood.
    
    Fetches messages in chunks of 100 to avoid timeouts.
    """
    messages = []
    parsed_offset_id = parse_message_id(offset_id) or 0
    total_limit = limit or (1 << 31) - 1  # Use max int if no limit
    chunk_size = min(100, total_limit)  # Get messages in chunks of 100
    current_offset = offset
    current_offset_id = parsed_offset_id
    
    while len(messages) < total_limit:
        try:
            # Calculate remaining messages to fetch
            remaining = total_limit - len(messages)
            current_chunk_size = min(chunk_size, remaining)
            
            # Get next chunk of messages
            chunk = []
            async for msg in mcp.client.get_chat_history(
                chat_id=chat_id,
                limit=current_chunk_size,
                offset=current_offset,
                offset_id=current_offset_id,
                **kwargs
            ):
                chunk.append(msg)
            
            if not chunk:  # No more messages
                break
            
            # Update offset for next chunk
            current_offset_id = chunk[-1].id
            messages.extend(chunk)
            
            # Break if we've reached the limit
            if len(messages) >= total_limit:
                break
            
        except FloodWait as e:
            await handle_flood_wait(e, "get_chat_history")
        except Exception as e:
            logger.error(f"Error in get_chat_history: {e}, chat_id={chat_id}, limit={limit}, offset={offset}, offset_id={offset_id}, kwargs={kwargs}")
            raise
    
    return messages[:total_limit]  # Ensure we don't return more than requested

async def search_messages_with_retry(chat_id: int | str, query: str, **kwargs):
    """Wrapper for search_messages with retry on flood.
    
    Fetches messages in chunks to avoid timeouts.
    """
    messages = []
    total_limit = kwargs.get('limit', 100)
    chunk_size = min(50, total_limit)  # Get messages in smaller chunks for search
    current_offset = 0
    
    while len(messages) < total_limit:
        try:
            # Calculate remaining messages to fetch
            remaining = total_limit - len(messages)
            current_chunk_size = min(chunk_size, remaining)
            
            # Get next chunk of messages
            chunk = []
            async for msg in mcp.client.search_messages(
                chat_id=chat_id,
                query=query,
                limit=current_chunk_size,
                offset=current_offset,
                **{k: v for k, v in kwargs.items() if k != 'limit'}  # Pass other kwargs except limit
            ):
                chunk.append(msg)
            
            if not chunk:  # No more messages
                break
            
            # Update offset for next chunk
            current_offset += len(chunk)
            messages.extend(chunk)
            
            # Break if we've reached the limit
            if len(messages) >= total_limit:
                break
            
        except FloodWait as e:
            await handle_flood_wait(e, "search_messages")
        except Exception as e:
            logger.error(f"Error in search_messages: {e}, chat_id={chat_id}, query={query}, kwargs={kwargs}")
            raise
    
    return messages[:total_limit]  # Ensure we don't return more than requested

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

def ensure_unicode(text: str | None) -> str:
    """Ensure text is properly encoded as Unicode."""
    if text is None:
        return ""
    return str(text)  # Simple conversion, let json.dumps handle encoding

def serialize_message(msg: Message) -> Dict[str, Any]:
    """Convert a Telegram message to a minimal JSON structure with proper Unicode handling."""
    data = {
        "id": msg.id,
        "date": msg.date.isoformat() if msg.date else None,
        # Properly handle text with potential Unicode characters
        "text": ensure_unicode(msg.text or msg.caption),
        "from": {
            "id": msg.from_user.id,
            "username": ensure_unicode(msg.from_user.username),
            "name": ensure_unicode(msg.from_user.first_name)
        } if msg.from_user else None,
        "chat": {
            "id": msg.chat.id,
            "title": ensure_unicode(msg.chat.title or msg.chat.first_name),
            "type": str(msg.chat.type)
        }
    }

    # Add optional fields only if present, ensuring Unicode for text fields
    if msg.edit_date:
        data["edited_at"] = msg.edit_date.isoformat()
    if msg.reply_to_message_id:
        data["reply_to"] = msg.reply_to_message_id
    if msg.forward_from:
        data["forwarded_from"] = {
            "id": msg.forward_from.id,
            "name": ensure_unicode(msg.forward_from.first_name)
        }
    if msg.views:
        data["views"] = msg.views
    
    # Add media info if present, ensuring Unicode for text fields
    if msg.media:
        if msg.document:
            data["media"] = {
                "type": "document",
                "name": ensure_unicode(msg.document.file_name),
                "size": msg.document.file_size
            }
        elif msg.photo:
            try:
                # Just use the largest photo size
                if isinstance(msg.photo, list):
                    photo = msg.photo[-1]
                    data["media"] = {
                        "type": "photo",
                        "width": photo.width,
                        "height": photo.height,
                        "size": photo.file_size
                    }
                else:
                    # Handle case where photo is not a list
                    photo = msg.photo
                    data["media"] = {
                        "type": "photo",
                        "width": getattr(photo, "width", 0),
                        "height": getattr(photo, "height", 0),
                        "size": getattr(photo, "file_size", 0)
                    }
            except Exception as e:
                logger.warning(f"Error processing message photo: {e}")
                data["media"] = {"type": "photo", "error": str(e)}
        else:
            data["media"] = {"type": ensure_unicode(str(msg.media))}
    
    return data

class TelegramBridge(FastMCP):
    def __init__(self):
        super().__init__("telegram-bridge")
        
        # Load environment variables
        load_dotenv()
        
        # Track client readiness
        self._client_started = False
        
        # Validate credentials
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.session_string = os.getenv("TELEGRAM_SESSION_STRING")

        if not all([self.api_id, self.api_hash, self.session_string]):
            raise ValueError(
                "Missing required environment variables. Please set TELEGRAM_API_ID, "
                "TELEGRAM_API_HASH, and TELEGRAM_SESSION_STRING"
            )

        # Set up XDG-compliant session directory
        session_dir = xdg_data_home() / "mcp_telegram" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using session directory: %s", session_dir)
        
        # Initialize Telegram client
        session_file = session_dir / "mcp_telegram.session"
        logger.debug("Session file path: %s", session_file)
        
        # First try loading from session file
        if session_file.exists():
            logger.info("Loading existing session from %s", session_file)
            self.client = Client(
                str(session_file.with_suffix("")),  # Pyrogram will add .session extension
                api_id=self.api_id,
                api_hash=self.api_hash,
                in_memory=False,
                no_updates=True  # Disable updates since we don't need them
            )
        else:
            # If no session file exists, start with temporary client
            logger.info("Creating new session at %s", session_file)
            
            # First create temporary client with session string
            temp_client = Client(
                name="temp",
                api_id=self.api_id,
                api_hash=self.api_hash,
                session_string=self.session_string,
                in_memory=True,
                no_updates=True
            )
            
            try:
                # Start temp client to get auth info
                logger.info("Starting temporary client...")
                async def migrate_session():
                    try:
                        # Start the client and wait for storage to be ready
                        await temp_client.start()
                        await temp_client.storage.open()
                        
                        # Create SQLite file
                        session_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy SQLite database from memory to file
                        memory_db = temp_client.storage.conn
                        file_db = sqlite3.connect(str(session_file))
                        logger.debug("Created SQLite file at %s", session_file)
                        
                        # Backup database
                        memory_db.backup(file_db)
                        logger.debug("Backed up session data from memory to file")
                        
                        # Close file connection
                        file_db.close()
                        logger.debug("Closed SQLite file connection")
                        
                        # Save any pending changes
                        await temp_client.storage.save()
                        logger.info("Session migration completed successfully")
                        
                    finally:
                        # Always stop the client
                        await temp_client.stop()
                
                # Run in event loop
                loop = asyncio.get_event_loop()
                loop.run_until_complete(migrate_session())
                
            except Exception as e:
                logger.error(f"Failed to migrate session: {e}")
                raise
            finally:
                # Clean up temp client
                del temp_client
            
            # Now create real client using the saved session file
            logger.info("Creating new client with migrated session file")
            self.client = Client(
                name=str(session_file.with_suffix("")),  # Full path without .session
                api_id=self.api_id,
                api_hash=self.api_hash,
                in_memory=False,
                no_updates=True
            )

    async def _start_client(self):
        """Start the Telegram client in the background."""
        try:
            logger.info("Starting Telegram client...")
            await retry_with_flood_handling(
                self.client.start,
                "start_client"
            )
            self._client_started = True
            logger.info("Telegram client started successfully")
        except FloodWait as e:
            logger.warning(f"Hit FloodWait while starting client, waiting {e.value} seconds")
            await asyncio.sleep(e.value)
            # Try again after waiting
            await self._start_client()
        except Exception as e:
            logger.error(f"Error starting Telegram client: {e}")
            # Don't raise the error - let the MCP server continue running

    def is_client_ready(self) -> bool:
        """Check if the Telegram client is ready for use."""
        return self._client_started and self.client.is_connected

    async def _run_with_transport(self, transport: Literal["stdio", "sse"]):
        """Internal method to run with specific transport"""
        # Start client in background
        client_task = asyncio.create_task(self._start_client())
        
        # Run MCP server and client concurrently
        try:
            if transport == "stdio":
                async with stdio_server() as (read_stream, write_stream):
                    await asyncio.gather(
                        client_task,
                        self._mcp_server.run(
                            read_stream,
                            write_stream,
                            self._mcp_server.create_initialization_options(),
                        )
                    )
            else:  # transport == "sse"
                await asyncio.gather(
                    client_task,
                    super().run_sse_async()
                )
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

async def handle_chat_errors(chat: str | int, ctx: Context | None, action: str):
    """Common error handling for chat operations.
    
    Args:
        chat: Chat identifier (string or integer)
        ctx: Context for error reporting
        action: Description of the action being performed (for logging)
    
    Returns:
        The resolved chat object
        
    Raises:
        Various Telegram exceptions with appropriate error messages
    """
    try:
        logger.debug(f"Resolving chat info for: {chat} (type: {type(chat)})")
        chat_obj = await retry_with_flood_handling(mcp.client.get_chat, "get_chat", chat)
        logger.debug(f"Chat resolved: id={chat_obj.id}, type={chat_obj.type}, raw_id={chat}")
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
        chat: Username, phone, or chat ID - the user might provide something else, like an alias or a name, you'd need to cross-check in another tool.
        limit: Maximum number of messages to return
        offset_id: Get messages before this ID
        from_date: Get messages from this date (ISO format)
        has_media: Filter messages with media
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})

    messages = []
    if not chat:
        raise ValueError("chat parameter is required")
    
    try:
        chat_obj = await handle_chat_errors(chat, ctx, "getting messages")
        logger.debug(f"Resolved chat object: id={chat_obj.id}, type={chat_obj.type}")
        
        date_filter = None
        if from_date:
            date_filter = datetime.fromisoformat(from_date)

        # Get messages with retry
        parsed_offset = parse_message_id(offset_id)
        logger.debug(f"Getting chat history with: id={chat_obj.id}, type={chat_obj.type}, limit={limit}, offset={parsed_offset}")
        
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
        result = json.dumps(messages, indent=2, ensure_ascii=False)
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
        query: Search query (simple strings, regex and QLs are not supported)
        chats: List of chats to search in
        limit: Maximum number of results
    """
    results = []
    chats = chats or ["me"]

    for chat in chats:
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

    return json.dumps(results, indent=2, ensure_ascii=False)

@mcp.tool()
async def get_full_chat_info(
    chat: str | None = None,
    ctx: Context | None = None
) -> str:
    """Get comprehensive information about a chat or user.
    
    Args:
        chat: Username, phone, or chat ID. Defaults to "me" for current user.
              The user might provide something else, like an alias or a name, you'd need to cross-check in another tool.
    
    Returns detailed information including:
    - Basic info (id, name/title, username)
    - Profile/chat photos
    - Bio/description
    - Phone number (for users)
    - Status (online/offline, verified, premium)
    - Member count (for groups/channels)
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})
    try:
        chat_obj = await handle_chat_errors(chat, ctx, "full chat info")
        logger.debug(f"Resolved chat object: id={chat_obj.id}, type={chat_obj.type}")
        
        # Get photos
        photos = []
        async for photo in mcp.client.get_chat_photos(chat_obj.id):
            photos.append({
                "id": photo.file_id,
                "unique_id": photo.file_unique_id,
                "width": photo.width,
                "height": photo.height,
                "size": photo.file_size,
                "date": photo.date.isoformat() if photo.date else None
            })
        
        # Build response based on chat type
        base_info = {
            "id": chat_obj.id,
            "type": str(chat_obj.type),
            "username": ensure_unicode(chat_obj.username) if chat_obj.username else None,
            "photos_count": len(photos),
            "photos": photos,
            "dc_id": chat_obj.dc_id if hasattr(chat_obj, "dc_id") else None,
            "has_protected_content": chat_obj.has_protected_content if hasattr(chat_obj, "has_protected_content") else False,
            "is_verified": chat_obj.is_verified if hasattr(chat_obj, "is_verified") else False
        }
        
        if str(chat_obj.type) in ["supergroup", "channel"]:
            info = {
                **base_info,
                "title": ensure_unicode(chat_obj.title),
                "description": ensure_unicode(chat_obj.description) if hasattr(chat_obj, "description") else None,
                "member_count": chat_obj.members_count if hasattr(chat_obj, "members_count") else None,
                "is_restricted": chat_obj.is_restricted if hasattr(chat_obj, "is_restricted") else False,
                "is_scam": chat_obj.is_scam if hasattr(chat_obj, "is_scam") else False,
                "is_fake": chat_obj.is_fake if hasattr(chat_obj, "is_fake") else False
            }
        else:  # user or bot
            info = {
                **base_info,
                "first_name": ensure_unicode(chat_obj.first_name),
                "last_name": ensure_unicode(chat_obj.last_name) if chat_obj.last_name else None,
                "phone_number": ensure_unicode(chat_obj.phone_number) if hasattr(chat_obj, "phone_number") else None,
                "bio": ensure_unicode(chat_obj.bio) if hasattr(chat_obj, "bio") else None,
                "is_bot": chat_obj.is_bot if hasattr(chat_obj, "is_bot") else False,
                "is_premium": chat_obj.is_premium if hasattr(chat_obj, "is_premium") else False,
                "language_code": chat_obj.language_code if hasattr(chat_obj, "language_code") else None
            }
        
        return json.dumps(info, indent=2, ensure_ascii=False)
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting info for chat {chat}: {e}")
        logger.error(f"Error getting info for chat {chat}: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
async def get_contacts(
    ctx: Context | None = None
) -> str:
    """Get contacts from your Telegram address book. Might be useful for looking people up."""
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})

    try:
        contacts = []
        for user in await mcp.client.get_contacts():
            contacts.append({
                "id": user.id,
                "username": ensure_unicode(user.username) if user.username else None,
                "first_name": ensure_unicode(user.first_name),
                "last_name": ensure_unicode(user.last_name) if user.last_name else None,
                "phone_number": ensure_unicode(user.phone_number) if hasattr(user, "phone_number") else None,
                "is_contact": True,
                "is_mutual_contact": user.is_mutual_contact if hasattr(user, "is_mutual_contact") else False,
                "is_deleted": user.is_deleted if hasattr(user, "is_deleted") else False,
                "is_bot": user.is_bot if hasattr(user, "is_bot") else False,
                "is_verified": user.is_verified if hasattr(user, "is_verified") else False,
                "language_code": user.language_code if hasattr(user, "language_code") else None
            })
            
            if ctx:
                await ctx.report_progress(len(contacts), len(contacts))
        
        return json.dumps(contacts, indent=2, ensure_ascii=False)
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting contacts: {e}")
        logger.error(f"Error getting contacts: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
async def search_in_chat(
    chat: str,
    query: str,
    limit: int = 100,
    ctx: Context | None = None
) -> str:
    """Search for messages in a specific chat.
    
    Args:
        chat: Username, phone, or chat ID - the user might provide something else, like an alias or a name, you'd need to cross-check in another tool.
        query: Search query (simple strings, regex and QLs are not supported)
        limit: Maximum number of messages to return
    
    Returns matching messages in JSON format.
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})
    try:
        chat_obj = await handle_chat_errors(chat, ctx, "search in chat")

        # Search messages with retry
        messages = []
        async for message in mcp.client.search_messages(
            chat_id=chat_obj.id,
            query=query,
            limit=limit
        ):
            messages.append(serialize_message(message))
            if ctx:
                await ctx.report_progress(len(messages), limit)
                ctx.info(f"Found message {len(messages)}/{limit}")
        
        return json.dumps(messages, indent=2, ensure_ascii=False)
    except Exception as e:
        if ctx:
            ctx.error(f"Error searching in chat {chat}: {e}")
        logger.error(f"Error searching in chat {chat}: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
async def get_chat_list(
    offset_id: str | int | None = None,
    offset_date: str | None = None,
    limit: int = 100,
    ctx: Context | None = None
) -> str:
    """Get list of chats with basic information. Might be useful for finding IDs.
    
    Args:
        offset_id: Get chats before this message ID
        offset_date: Get chats before this date (ISO format)
        limit: Maximum number of chats to return (default: 100)
    
    Returns a list of chats with:
    - Basic info (id, title/name, type)
    - Member count (for groups/channels)
    - Latest message
    - Chat photo info
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})
    try:
        chats = []
        total_retrieved = 0
        
        # Convert offset_date to timestamp if provided
        offset_date_ts = 0
        if offset_date:
            offset_date_ts = int(datetime.fromisoformat(offset_date).timestamp())
        
        # Parse and convert offset_id
        offset_id_int = parse_message_id(offset_id) or 0
        logger.debug(f"Parsed offset ID: {offset_id_int}")
        
        # Convert channel ID if needed
        if isinstance(offset_id_int, int) and offset_id_int < 0:
            # For channel IDs, we don't need any special handling
            logger.debug(f"Using channel ID for get_chat_list offset: {offset_id_int}")
        
        # Get dialogs with offset and limit
        logger.debug(f"Getting dialogs with: limit={limit}, offset_id={offset_id_int}, offset_date={offset_date_ts}")
        async for dialog in mcp.client.get_dialogs(limit=limit):
            # Skip until we reach the offset_id if provided
            if offset_id_int and dialog.top_message and dialog.top_message.id >= offset_id_int:
                continue
            # Skip until we reach the offset_date if provided
            if offset_date_ts and dialog.top_message and dialog.top_message.date.timestamp() >= offset_date_ts:
                continue
            
            chat = dialog.chat
            
            # Build basic info
            info = {
                "id": chat.id,
                "type": str(chat.type),
                "title": ensure_unicode(chat.title) if hasattr(chat, "title") else None,
                "first_name": ensure_unicode(chat.first_name) if hasattr(chat, "first_name") else None,
                "last_name": ensure_unicode(chat.last_name) if hasattr(chat, "last_name") else None,
                "username": ensure_unicode(chat.username) if hasattr(chat, "username") else None,
                
                # Group/channel specific
                "member_count": chat.members_count if hasattr(chat, "members_count") else None,
                "is_verified": chat.is_verified if hasattr(chat, "is_verified") else None,
                "is_restricted": chat.is_restricted if hasattr(chat, "is_restricted") else None,
                
                # Latest message if available
                "latest_message": serialize_message(dialog.top_message) if dialog.top_message else None
            }
            
            # Add photo info if available
            if hasattr(chat, "photo") and chat.photo is not None:
                photo = chat.photo
                from pyrogram.types.user_and_chats.chat_photo import ChatPhoto
                if isinstance(photo, ChatPhoto):
                    info["photo"] = {
                        "small_file_id": photo.small_file_id,
                        "small_photo_unique_id": photo.small_photo_unique_id,
                        "big_file_id": photo.big_file_id,
                        "big_photo_unique_id": photo.big_photo_unique_id
                    }
                else:
                    info["photo"] = None
            else:
                info["photo"] = None
            
            chats.append(info)
            total_retrieved += 1
            
            if ctx:
                await ctx.report_progress(total_retrieved, limit)
                ctx.info(f"Retrieved chat {total_retrieved}/{limit}")
            
            if total_retrieved >= limit:
                break
        
        return json.dumps(chats, indent=2, ensure_ascii=False)
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting chat list: {e}")
        logger.error(f"Error getting chat list: {e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
async def get_media_content(
    file_id: str,
    ctx: Context | None = None
) -> MCPImage:
    """Get media content using a file_id.
    
    Args:
        file_id: The Telegram file_id to retrieve
    
    Returns the media content as an image that Claude can view.
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        raise RuntimeError(error)

    try:
        # Download the file
        file = await mcp.client.download_media(file_id, in_memory=True)
        
        if not file:
            error = "Failed to download media"
            if ctx:
                ctx.error(error)
            logger.error(error)
            raise RuntimeError(error)

        # Process with PIL to ensure format and size constraints
        if isinstance(file, io.BytesIO):
            file_data = file.getvalue()
        else:
            file_data = file
            
        image = PILImage.open(io.BytesIO(file_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 1000x1000)
        max_size = 1000
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size/image.width, max_size/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, PILImage.Resampling.LANCZOS)

        # Save as JPEG with compression
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=60, optimize=True)
        
        # Return as MCP Image type
        return MCPImage(data=output.getvalue(), format="jpeg")

    except Exception as e:
        error = f"Error getting media content: {str(e)}"
        if ctx:
            ctx.error(error)
        logger.error(error)
        raise RuntimeError(error)

@mcp.tool()
async def send_message(
    chat: str,
    text: str,
    reply_to: str | int | None = None,
    parse_mode: Literal["markdown", "html", "disabled"] = "markdown",
    disable_web_page_preview: bool = True,
    disable_notification: bool = False,
    ctx: Context | None = None
) -> str:
    """Send a message to a chat. You may want to add a disclaimer to the message, that it was sent automatically, if the recipient is not a bot.
    
    Args:
        chat: Username, phone, or chat ID - the user might provide something else, like an alias or a name, you'd need to cross-check in another tool.
        text: The message text to send. Supports formatting based on parse_mode.
        reply_to: Message ID to reply to (optional)
        parse_mode: Text formatting mode:
            - "markdown": Use Markdown V2 formatting (default)
            - "html": Use HTML formatting
            - "disabled": No formatting
        disable_web_page_preview: Whether to disable web page previews (default: True)
        disable_notification: Whether to send the message silently (default: False)
    
    Returns:
        JSON string containing the sent message info
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})
    
    try:
        # Parse chat ID and get chat info
        chat_obj = await handle_chat_errors(chat, ctx, "sending message")
        
        # Parse reply_to message ID if provided
        reply_to_id = parse_message_id(reply_to)
        
        # Convert parse mode to Pyrogram enum
        if parse_mode == "markdown":
            mode = enums.ParseMode.MARKDOWN
        elif parse_mode == "html":
            mode = enums.ParseMode.HTML
        else:
            mode = None
        
        # Send message with retry
        async def send():
            return await mcp.client.send_message(
                chat_id=chat_obj.id,
                text=text,
                parse_mode=mode,
                disable_web_page_preview=disable_web_page_preview,
                disable_notification=disable_notification,
                reply_to_message_id=reply_to_id
            )
        
        message = await retry_with_flood_handling(send, "send_message")
        
        if ctx:
            ctx.info(f"Message sent successfully to {chat}")
        
        # Return serialized message info
        return json.dumps(serialize_message(message), indent=2, ensure_ascii=False)
    
    except ChatWriteForbidden as e:
        error = f"Cannot send messages to {chat}. You don't have permission to write in this chat."
        if ctx:
            ctx.error(error)
        logger.error(f"Write forbidden while sending message to {chat}: {e}")
        return json.dumps({"error": error})
    except Exception as e:
        error = f"Error sending message to {chat}: {str(e)}"
        if ctx:
            ctx.error(error)
        logger.error(error)
        return json.dumps({"error": error})

@mcp.tool()
async def get_common_chats(
    user: str,
    ctx: Context | None = None
) -> str:
    """Get common chats with a user.
    
    Args:
        user: Username, phone, or user ID - the user might provide something else, like an alias or a name, you'd need to cross-check in another tool.
    
    Returns a list of chats that both you and the user are members of.
    Raises ValueError if the ID doesn't belong to a user.
    """
    if not mcp.is_client_ready():
        error = "Telegram client is still starting up, please try again in a few seconds"
        if ctx:
            ctx.error(error)
        logger.warning(error)
        return json.dumps({"error": error})
    try:
        user_obj = await handle_chat_errors(user, ctx, "get common chats")
                
        common_chats = []
        
        for chat in await mcp.client.get_common_chats(user_obj.id):
            info = {
                "id": chat.id,
                "type": str(chat.type),
                "title": ensure_unicode(chat.title) if hasattr(chat, "title") else None,
                "username": ensure_unicode(chat.username) if hasattr(chat, "username") else None,
                "member_count": chat.members_count if hasattr(chat, "member_count") else None,
                "is_verified": chat.is_verified if hasattr(chat, "is_verified") else None,
                "is_restricted": chat.is_restricted if hasattr(chat, "is_restricted") else None,
                "has_protected_content": chat.has_protected_content if hasattr(chat, "has_protected_content") else False
            }
            
            # Add photo info if available
            try:
                if hasattr(chat, "photo") and chat.photo is not None:
                    photo = chat.photo
                    if hasattr(photo, "__getitem__"):
                        try:
                            # Try to access as list
                            photo_item = photo[-1]
                            info["photo"] = {
                                "small_file_id": photo_item.file_id,
                                "small_photo_unique_id": photo_item.file_unique_id,
                                "big_file_id": photo_item.file_id,
                                "big_photo_unique_id": photo_item.file_unique_id
                            }
                        except (IndexError, TypeError, AttributeError):
                            # Fall back to treating as single object
                            if hasattr(photo, "small_file_id"):
                                info["photo"] = {
                                    "small_file_id": photo.small_file_id,
                                    "small_photo_unique_id": photo.small_photo_unique_id,
                                    "big_file_id": photo.big_file_id,
                                    "big_photo_unique_id": photo.big_photo_unique_id
                                }
                            else:
                                info["photo"] = None
                    else:
                        # Handle as single object
                        if hasattr(photo, "small_file_id"):
                            info["photo"] = {
                                "small_file_id": photo.small_file_id,
                                "small_photo_unique_id": photo.small_photo_unique_id,
                                "big_file_id": photo.big_file_id,
                                "big_photo_unique_id": photo.big_photo_unique_id
                            }
                        else:
                            info["photo"] = None
            except Exception as e:
                logger.warning(f"Error processing chat photo in get_common_chats: {e}")
                info["photo"] = None
            
            common_chats.append(info)
            
            if ctx:
                await ctx.report_progress(len(common_chats), len(common_chats))
        
        return json.dumps(common_chats, indent=2, ensure_ascii=False)
    except ValueError as e:
        error = str(e)
        if ctx:
            ctx.error(error)
        logger.error(error)
        return json.dumps({"error": error})
    except Exception as e:
        if ctx:
            ctx.error(f"Error getting common chats with {user}: {e}")
        logger.error(f"Error getting common chats with {user}: {e}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    try:
        # Run the MCP server
        logger.info("Starting MCP server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
