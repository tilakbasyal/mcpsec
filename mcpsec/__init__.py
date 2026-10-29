import logging

# The MCP SDK logs every request at INFO level; keep the output readable.
for _name in ("mcp", "mcp.server", "mcp.server.lowlevel.server", "httpx", "openai", "openai._base_client"):
    logging.getLogger(_name).setLevel(logging.WARNING)
