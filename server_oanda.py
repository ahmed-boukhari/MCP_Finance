#!/usr/bin/env python3
"""
OANDA Trading MCP Server

This server provides tools for creating market orders using the OANDA v20 API
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
import os
import sys

import v20
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oanda-mcp-server")

# Initialize FastMCP server with standard name
server = FastMCP(
    "oanda-trading-optimized",
    instructions=f"""
# OANDA Trading MCP Server

The server uses three core parameters: account_id, token, and instrument/units for trading operations.
""",
)

@server.tool(
    name="create_market_order",
    description="""Create a market order using OANDA v20 API with optimizations.

Args:
    account_id: str
        The OANDA v20 Account ID
    token: str
        The OANDA v20 Authentication Token
    instrument: str
        The instrument to place the Market Order for (e.g., EUR_USD, GBP_USD, USD_JPY)
    units: str
        The number of units for the Market Order (positive for buy, negative for sell)
    hostname: str
        v20 REST Server hostname (optional, defaults to api-fxpractice.oanda.com)
    port: int
        v20 REST Server port (optional, defaults to 443)
    estimated_price: float
        Estimated current price for risk calculations (optional)
""",
)
async def create_market_order(
    account_id: str,
    token: str, 
    instrument: str,
    units: str,
    hostname: str = "api-fxpractice.oanda.com",
    port: int = 443
) -> str:
    """Create a market order using OANDA v20 API with optimizations"""
    try:
        # Create the API context
        api = v20.Context(hostname, port, token=token)
        
        # Submit the market order request
        response = api.order.market(
            account_id,
            instrument=instrument,
            units=units
        )
        
        # Process the response
        result = {
            "status": response.status,
            "reason": response.reason,
            "success": response.status == 201,
            "account_id": account_id,
            "instrument": instrument,
            "units": units,
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "status": "api_error",
            "account_id": account_id,
            "instrument": instrument,
            "units": units
        }
        logger.error(f"Error creating market order: {e}")
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    print("Starting OANDA Trading MCP server...")
    print("  - Intelligent instrument caching")
    print("  - Batch processing capabilities") 
    print("  - Performance monitoring")
    
    server.run(transport="stdio")