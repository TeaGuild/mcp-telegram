# Telegram MCP

A Model Context Protocol server that provides structured access to Telegram data. This MCP acts as a data bridge, allowing AI assistants to efficiently access and analyze Telegram messages, media, and chat information.

## Setup

1. Get your Telegram API credentials:
   ```bash
   # 1. Go to https://my.telegram.org/auth
   # 2. Log in with your phone number
   # 3. Go to 'API development tools'
   # 4. Create a new application
   # 5. Note down your api_id and api_hash
   ```

2. Create a `.env` file:
   ```bash
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   ```

3. Generate a session string:
   ```bash
   python scripts/get_session.py
   ```
   This will prompt for your phone number and verification code, then generate a session string to add to your `.env` file.

4. Install dependencies:
   ```bash
   uv pip install -e .
   ```

5. Run the MCP server:
   ```bash
   mcp dev mcp_telegram/server.py
   ```

## Available Tools

### get_messages
Get messages from a chat with optional filtering.

```python
result = await mcp.call_tool("get_messages", {
    "chat": "@username",
    "limit": 10,
    "from_date": "2024-02-01T00:00:00",
    "has_media": True
})
```

Returns structured message data including:
- Message text and timestamps
- Sender information
- Chat context
- Media details
- Forward information
- View counts

### search_messages
Search for messages across chats.

```python
result = await mcp.call_tool("search_messages", {
    "query": "project meeting",
    "chats": ["@team", "@project"],
    "limit": 10
})
```

### get_chat_info
Get detailed information about a chat.

```python
result = await mcp.call_tool("get_chat_info", {
    "chat": "@username"
})
```

## Security Notes

- The session string provides full access to your Telegram account
- Store credentials securely in environment variables
- The MCP operates with your user account permissions

## Development

Run the server with debug logging:
```bash
mcp dev mcp_telegram/server.py
```

Logs are written to `logs/telegram_server.log`.
