# Telegram MCP

A Model Context Protocol (MCP) server that provides structured access to Telegram data. This MCP acts as a data bridge, allowing AI assistants to efficiently access and analyze Telegram messages, media, and chat information through the [Claude MCP platform](https://docs.anthropic.com/en/docs/agents-and-tools/mcps/using-mcps).

## Setup

1. Get your Telegram API credentials:
   ```bash
   # 1. Go to https://my.telegram.org/auth
   # 2. Log in with your phone number
   # 3. Go to 'API development tools'
   # 4. Create a new application
   # 5. Note down your api_id and api_hash
   ```

2. Clone this repository and create a `.env` file:
   ```bash
   git clone https://github.com/yourusername/mcp-telegram.git
   cd mcp-telegram
   cp .env.example .env
   # Edit .env to add your API credentials
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   # Or if you use uv:
   uv pip install -e .
   ```

4. Generate a session string:
   ```bash
   python scripts/get_session.py
   ```
   This will prompt for your phone number and verification code, then generate a session string to add to your `.env` file.

5. Run the MCP server:
   ```bash
   mcp dev mcp_telegram/server.py
   ```

   The server logs are stored in `logs/telegram_server.log`.

## Available Tools

This MCP provides these tools for interacting with Telegram:

### get_messages
Get messages from a chat with optional filtering.

```python
result = await mcp.call_tool("get_messages", {
    "chat": "@username",
    "limit": 10,
    "offset_id": "12345",  # Get messages before this ID
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

### get_full_chat_info
Get comprehensive information about a chat or user.

```python
result = await mcp.call_tool("get_full_chat_info", {
    "chat": "@username"  # Defaults to "me" for your own account
})
```

### get_chat_list
Get a list of chats (conversations) with basic information.

```python
result = await mcp.call_tool("get_chat_list", {
    "limit": 100,  # Maximum number of chats to return
    "offset_id": "12345",  # Get chats before this message ID
    "offset_date": "2024-02-01T00:00:00"  # Get chats before this date
})
```

### get_contacts
Get contacts from your Telegram address book.

```python
result = await mcp.call_tool("get_contacts")
```

### send_message
Send a message to a chat (with appropriate disclaimer).

```python
result = await mcp.call_tool("send_message", {
    "chat": "@username",
    "text": "Hello from the Telegram MCP! This message was sent by an AI assistant.",
    "parse_mode": "markdown",  # Can be "markdown", "html", or "disabled"
    "disable_web_page_preview": True
})
```

### get_media_content
Get media content (images) using a file_id.

```python
result = await mcp.call_tool("get_media_content", {
    "file_id": "ABCDEFG..."  # File ID obtained from message media
})
```

## Security Notes

- **Important**: The session string provides full access to your Telegram account. Treat it like a password.
- Store credentials securely in environment variables and never commit them to version control.
- The MCP operates with your user account permissions - it can access any chat you have joined.
- Review the messages being sent with the `send_message` tool to ensure they comply with Telegram's terms of service.

## Development

Run the server with debug logging:
```bash
mcp dev mcp_telegram/server.py
```

Logs are written to `logs/telegram_server.log`.

## Integration with Claude

To use this MCP with Claude:

1. Start the MCP server
2. In a separate terminal, run:
   ```bash
   mcp client --claude
   ```
3. Connect to the MCP in the Claude conversation:
   ```
   /connect telegram
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
