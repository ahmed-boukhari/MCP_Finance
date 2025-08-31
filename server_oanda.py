#!/usr/bin/env python3
"""
OANDA Trading MCP Server

This server provides tools for creating market orders using the OANDA v20 API.
"""

import json
import logging
from enum import Enum
import v20
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oanda-mcp-server")

# Try to import common modules, provide fallbacks if not available
try:
    import common.args
    from order.view import print_order_create_response_transactions
except ImportError as e:
    logger.warning(f"Could not import common modules: {e}")
    # Fallback implementations if the modules aren't available
    class CommonArgs:
        @staticmethod
        def instrument(value):
            return str(value).upper().replace("/", "_")
    
    common = type('obj', (object,), {'args': CommonArgs()})()
    
    def print_order_create_response_transactions(response):
        """Fallback function to print order response"""
        if hasattr(response, 'body') and response.body:
            if hasattr(response.body, 'orderCreateTransaction'):
                logger.info(f"Order Create Transaction: {response.body.orderCreateTransaction}")
            if hasattr(response.body, 'orderFillTransaction'):
                logger.info(f"Order Fill Transaction: {response.body.orderFillTransaction}")

# Initialize FastMCP server
oanda_server = FastMCP(
    "oanda-trading",
    instructions="""
# OANDA Trading MCP Server

This server provides tools for creating market orders using the OANDA v20 API.

Available tools:
- create_market_order: Create a market order using OANDA v20 API. Supports both practice and live trading environments.

The server requires OANDA v20 account credentials (account ID and authentication token) to execute trades.
For practice trading, use hostname 'api-fxpractice.oanda.com'. For live trading, use 'api-fxtrade.oanda.com'.
""",
)


@oanda_server.tool(
    name="create_market_order",
    description="""Create a market order using OANDA v20 API.

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
        v20 REST Server hostname (optional, defaults to api-fxpractice.oanda.com for practice trading)
    port: int
        v20 REST Server port (optional, defaults to 443)
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
    """Create a market order using OANDA v20 API
    
    Args:
        account_id: The OANDA v20 Account ID
        token: The OANDA v20 Authentication Token  
        instrument: The instrument to place the Market Order for
        units: The number of units for the Market Order
        hostname: v20 REST Server hostname (defaults to practice server)
        port: v20 REST Server port (defaults to 443)
    """
    try:
        # Create the API context
        api = v20.Context(
            hostname,
            port,
            token=token
        )
        
        # Validate and format instrument
        try:
            if hasattr(common, 'args') and hasattr(common.args, 'instrument'):
                formatted_instrument = common.args.instrument(instrument)
            else:
                # Fallback formatting
                formatted_instrument = str(instrument).upper().replace("/", "_")
        except Exception as e:
            logger.warning(f"Could not validate instrument: {e}")
            formatted_instrument = str(instrument).upper().replace("/", "_")
        
        # Submit the request to create the Market Order
        response = api.order.market(
            account_id,
            instrument=formatted_instrument,
            units=units
        )
        
        # Process the response
        result = {
            "status": response.status,
            "reason": response.reason,
            "success": response.status == 201,
            "account_id": account_id,
            "instrument": formatted_instrument,
            "units": units
        }
        
        # Add transaction details if available
        if hasattr(response, 'body') and response.body:
            body_dict = {}
            
            if hasattr(response.body, 'orderCreateTransaction'):
                transaction = response.body.orderCreateTransaction
                if hasattr(transaction, 'dict'):
                    body_dict["orderCreateTransaction"] = transaction.dict()
                elif hasattr(transaction, '__dict__'):
                    body_dict["orderCreateTransaction"] = transaction.__dict__
                else:
                    body_dict["orderCreateTransaction"] = str(transaction)
            
            if hasattr(response.body, 'orderFillTransaction'):
                fill_transaction = response.body.orderFillTransaction
                if hasattr(fill_transaction, 'dict'):
                    body_dict["orderFillTransaction"] = fill_transaction.dict()
                elif hasattr(fill_transaction, '__dict__'):
                    body_dict["orderFillTransaction"] = fill_transaction.__dict__
                else:
                    body_dict["orderFillTransaction"] = str(fill_transaction)
            
            if hasattr(response.body, 'orderCancelTransaction'):
                cancel_transaction = response.body.orderCancelTransaction
                if hasattr(cancel_transaction, 'dict'):
                    body_dict["orderCancelTransaction"] = cancel_transaction.dict()
                elif hasattr(cancel_transaction, '__dict__'):
                    body_dict["orderCancelTransaction"] = cancel_transaction.__dict__
                else:
                    body_dict["orderCancelTransaction"] = str(cancel_transaction)
            
            if hasattr(response.body, 'lastTransactionID'):
                body_dict["lastTransactionID"] = response.body.lastTransactionID
            
            if hasattr(response.body, 'relatedTransactionIDs'):
                body_dict["relatedTransactionIDs"] = response.body.relatedTransactionIDs
            
            result.update(body_dict)
        
        # Log the transaction details using the imported function
        try:
            print_order_create_response_transactions(response)
        except Exception as e:
            logger.warning(f"Could not print transaction details: {e}")
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "status": "error",
            "account_id": account_id,
            "instrument": instrument,
            "units": units
        }
        logger.error(f"Error creating market order: {e}")
        return json.dumps(error_result, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting OANDA Trading MCP server...")
    oanda_server.run(transport="stdio")