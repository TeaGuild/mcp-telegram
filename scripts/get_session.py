#!/usr/bin/env python3
"""
Helper script to generate a Telegram userbot session string using Pyrogram.
This script will use your API credentials from .env and prompt for your phone number
to create a session string that can be used in the MCP server.
"""

import os
import asyncio
from pyrogram import Client
from dotenv import load_dotenv

load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

if not API_ID or not API_HASH:
    raise ValueError(
        "Please make sure TELEGRAM_API_ID and TELEGRAM_API_HASH are set in your .env file"
    )


async def main():
    print("\nCreating Telegram userbot session...")
    print("This will log you in as a user account (userbot)")
    print()

    # Create client
    async with Client("mcp_telegram", api_id=API_ID, api_hash=API_HASH, in_memory=True) as client:
        # Get and print session string
        session_string = await client.export_session_string()
        print("\nYour userbot session string (save this securely):\n")
        print(session_string)
        print("\nAdd this to your .env file as TELEGRAM_SESSION_STRING")


if __name__ == "__main__":
    asyncio.run(main())
