# MCP Telegram Bridge Architecture

## Overview
The MCP Telegram Bridge provides a bridge between MCP (Message Control Protocol) and Telegram's API through Pyrogram. It enables applications to interact with Telegram through a standardized interface.

## Components

### FastMCP Server
- Handles MCP protocol communication
- Manages tools and resources
- Provides error handling and progress reporting

### Pyrogram Client
- Manages Telegram API communication
- Handles authentication and session management
- Provides low-level API access

## Tools

### get_messages
- Retrieves messages from chats
- Supports filtering by date and media
- Handles pagination through offset_id
- Returns messages in a standardized JSON format
- Includes error handling for rate limits and access control

### search_messages  
- Searches for messages across chats
- Supports query-based searching
- Handles rate limiting and retries
- Can search across multiple chats simultaneously
- Returns standardized message format

### get_chat_info
- Retrieves detailed chat information
- Supports different chat types (user, group, channel)
- Handles access control checks
- Returns comprehensive chat metadata
- Includes error handling for private/restricted chats

## Resources
Resources are used for caching frequently accessed data that rarely changes.

### resource://telegram/peer/{peer_id}
- Caches peer information from Pyrogram's storage
- Includes access_hash, type, username, phone_number
- Critical for many API operations
- TTL: 8 hours for usernames
- Used for optimizing API calls

### resource://telegram/session
- Caches current session information
- Includes dc_id, user_id, is_bot, auth status
- Required for maintaining connection
- Persists for session duration
- Core component for API authentication

### resource://telegram/me
- Caches current user information
- Frequently accessed in operations
- Changes only on re-login
- Essential for user context

## Error Handling
- Comprehensive error handling for Telegram API errors
  - UsernameNotOccupied
  - PeerIdInvalid
  - ChannelInvalid
  - ChatAdminRequired
  - UserNotParticipant
  - FloodWait
  - SlowmodeWait
  - ChatWriteForbidden
  - ChannelPrivate
  - UserDeactivated
  - UserDeactivatedBan
- Rate limit handling with exponential backoff
- Access control validation
- Progress reporting through MCP context

## Session Management
- Uses Pyrogram's session system
- Supports both memory and file-based storage
- Handles session string import/export
- Manages authentication flow
- Maintains persistent connections
- Handles connection recovery

## Implementation Details

### Message Serialization
- Messages are serialized to a consistent JSON format
- Handles various message types (text, media, etc.)
- Preserves essential metadata
- Ensures proper Unicode handling
- Includes message context (reply, forward info)

### Rate Limiting
- Implements exponential backoff for FloodWait
- Handles per-chat slowmode restrictions
- Manages global rate limits
- Provides retry mechanisms
- Reports progress during delays

### Caching Strategy
- Uses Pyrogram's built-in storage system
- Caches frequently accessed peer information
- Maintains session data
- Optimizes API calls
- Reduces unnecessary network requests
