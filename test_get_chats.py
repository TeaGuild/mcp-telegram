import asyncio
from mcp_telegram.server import mcp


async def main():
    # Start the client
    await mcp._start_client()

    # Wait a bit for client to be ready
    while not mcp.is_client_ready():
        print("Waiting for client to be ready...")
        await asyncio.sleep(1)

    try:
        # Call get_chat_list directly as it's decorated with @mcp.tool()
        result = await mcp.get_chat_list()
        print(result)
    finally:
        # Clean up
        if mcp.client.is_connected:
            await mcp.client.stop()


if __name__ == "__main__":
    asyncio.run(main())
