#!/usr/bin/env python3
"""
LLVM-Optimized OANDA Trading MCP Server

This server provides tools for creating market orders using the OANDA v20 API
with performance optimizations using LLVM via Numba.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
import os
import sys

# Fix Numba/LLVM import issues
try:
    # Set Numba environment variables before import
    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
    os.environ['NUMBA_DISABLE_HSA'] = '1'
    os.environ['NUMBA_DISABLE_CUDA'] = '1'
    os.environ['NUMBA_THREADING_LAYER'] = 'safe'
    
    # Try importing numba with error handling
    import numba
    from numba import njit, types
    from numba.core import config
    
    # Configure Numba for stability
    config.THREADING_LAYER = 'safe'
    
    NUMBA_AVAILABLE = True
    print(f"Numba version: {numba.__version__}")
    print(f"LLVM version: {numba.llvmlite.llvmlite.get_version()}")
    
except ImportError as e:
    print(f"Numba not available: {e}")
    print("Install with: pip install numba")
    NUMBA_AVAILABLE = False
except Exception as e:
    print(f"Numba configuration error: {e}")
    print("Falling back to non-LLVM mode")
    NUMBA_AVAILABLE = False

import v20
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oanda-mcp-server")

# Conditional LLVM-optimized functions
if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=True)
    def validate_instrument_format_llvm(instrument_chars: np.ndarray) -> bool:
        """
        LLVM-optimized instrument validation.
        Checks if instrument follows proper forex pair format (e.g., EUR_USD).
        """
        length = len(instrument_chars)
        
        # Must be at least 7 characters (XXX_YYY)
        if length < 7:
            return False
        
        # Check for underscore separator at position 3
        if instrument_chars[3] != 95:  # ord('_') = 95
            return False
        
        # Validate currency code format (3 letters each side)
        for i in range(3):  # First currency
            char_code = instrument_chars[i]
            if not (65 <= char_code <= 90):  # A-Z
                return False
        
        for i in range(4, min(7, length)):  # Second currency
            char_code = instrument_chars[i]
            if not (65 <= char_code <= 90):  # A-Z
                return False
        
        return length == 7  # Exact length check

    @njit(cache=True, nogil=True, fastmath=True)
    def format_instrument_llvm(instrument_chars: np.ndarray) -> np.ndarray:
        """
        LLVM-optimized instrument formatting.
        Converts instrument to uppercase and replaces '/' with '_'.
        """
        result = np.empty_like(instrument_chars)
        
        for i in range(len(instrument_chars)):
            char = instrument_chars[i]
            
            # Convert to uppercase
            if 97 <= char <= 122:  # a-z to A-Z
                result[i] = char - 32
            elif char == 47:  # ord('/') = 47, replace with '_'
                result[i] = 95  # ord('_') = 95
            else:
                result[i] = char
        
        return result

    @njit(cache=True, nogil=True, fastmath=True)
    def validate_units_llvm(units_value: float) -> tuple:
        """
        LLVM-optimized units validation.
        Returns (is_valid, error_code) where error_code:
        0 = valid, 1 = zero, 2 = too_small, 3 = too_large
        """
        if units_value == 0.0:
            return (False, 1)
        
        abs_units = abs(units_value)
        
        if abs_units < 1.0:
            return (False, 2)
        
        # Maximum position size check
        if abs_units > 10000000.0:  # 10M units
            return (False, 3)
        
        return (True, 0)

    @njit(cache=True, nogil=True, fastmath=True)
    def calculate_margin_requirement_llvm(units: float, estimated_price: float) -> float:
        """
        LLVM-optimized margin calculation.
        Standard forex margin requirement (2% for major pairs).
        """
        abs_units = abs(units)
        notional_value = abs_units * estimated_price
        return notional_value * 0.02

    @njit(cache=True, nogil=True, fastmath=True)
    def batch_validate_orders_llvm(units_array: np.ndarray, prices_array: np.ndarray, 
                                  max_margin: float) -> np.ndarray:
        """
        LLVM-optimized batch validation for multiple orders.
        Returns boolean array indicating which orders are valid.
        """
        n_orders = len(units_array)
        valid_flags = np.zeros(n_orders, dtype=np.bool_)
        total_margin = 0.0
        
        for i in range(n_orders):
            units = units_array[i]
            price = prices_array[i]
            
            # Individual validation
            is_valid, error_code = validate_units_llvm(units)
            if not is_valid:
                continue
            
            # Calculate margin for this order
            margin_required = calculate_margin_requirement_llvm(units, price)
            
            # Check if adding this order would exceed total margin
            if total_margin + margin_required <= max_margin:
                valid_flags[i] = True
                total_margin += margin_required
        
        return valid_flags

else:
    # Fallback implementations without Numba
    def validate_instrument_format_llvm(instrument_chars):
        """Fallback instrument validation."""
        instrument = ''.join(chr(int(c)) for c in instrument_chars)
        if len(instrument) != 7:
            return False
        if instrument[3] != '_':
            return False
        return (instrument[:3].isupper() and instrument[:3].isalpha() and 
                instrument[4:7].isupper() and instrument[4:7].isalpha())

    def format_instrument_llvm(instrument_chars):
        """Fallback instrument formatting."""
        result = []
        for char in instrument_chars:
            c = int(char)
            if 97 <= c <= 122:  # a-z to A-Z
                result.append(c - 32)
            elif c == 47:  # '/' to '_'
                result.append(95)
            else:
                result.append(c)
        return np.array(result)

    def validate_units_llvm(units_value):
        """Fallback units validation."""
        if units_value == 0.0:
            return (False, 1)
        abs_units = abs(units_value)
        if abs_units < 1.0:
            return (False, 2)
        if abs_units > 10000000.0:
            return (False, 3)
        return (True, 0)

    def calculate_margin_requirement_llvm(units, estimated_price):
        """Fallback margin calculation."""
        abs_units = abs(units)
        notional_value = abs_units * estimated_price
        return notional_value * 0.02

    def batch_validate_orders_llvm(units_array, prices_array, max_margin):
        """Fallback batch validation."""
        n_orders = len(units_array)
        valid_flags = np.zeros(n_orders, dtype=bool)
        total_margin = 0.0
        
        for i in range(n_orders):
            units = units_array[i]
            price = prices_array[i]
            
            is_valid, _ = validate_units_llvm(units)
            if not is_valid:
                continue
            
            margin_required = calculate_margin_requirement_llvm(units, price)
            
            if total_margin + margin_required <= max_margin:
                valid_flags[i] = True
                total_margin += margin_required
        
        return valid_flags

# Optimized caching system with thread safety
class InstrumentCache:
    """Thread-safe caching system for instrument formatting."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self._lock = None
        
        # Initialize thread lock if available
        try:
            import threading
            self._lock = threading.Lock()
        except ImportError:
            pass  # Single-threaded fallback
    
    def _with_lock(self, func):
        """Execute function with lock if available."""
        if self._lock:
            with self._lock:
                return func()
        else:
            return func()
    
    def get_formatted_instrument(self, instrument: str) -> Optional[str]:
        """Get cached formatted instrument or compute and cache it."""
        def _get_cached():
            if instrument in self.cache:
                return self.cache[instrument]
            return None
        
        # Check cache first
        cached_result = self._with_lock(_get_cached)
        if cached_result is not None:
            return cached_result
        
        # Process instrument
        try:
            # Convert to numpy array for LLVM processing
            instrument_chars = np.array([ord(c) for c in instrument], dtype=np.int32)
            
            # Format using LLVM
            formatted_chars = format_instrument_llvm(instrument_chars)
            formatted_instrument = ''.join(chr(int(c)) for c in formatted_chars)
            
            # Validate using LLVM
            if not validate_instrument_format_llvm(formatted_chars):
                return None
            
            # Cache the result
            def _cache_result():
                if len(self.cache) >= self.max_size:
                    oldest = self.access_order.pop(0)
                    del self.cache[oldest]
                
                self.cache[instrument] = formatted_instrument
                self.access_order.append(instrument)
            
            self._with_lock(_cache_result)
            return formatted_instrument
            
        except Exception as e:
            logger.error(f"Error processing instrument {instrument}: {e}")
            return None

# Global cache instance
instrument_cache = InstrumentCache()

# Performance monitoring decorator
def performance_monitor(func):
    """Decorator to monitor function performance."""
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed in {execution_time:.4f}s: {e}")
            raise
    return wrapper

# Initialize FastMCP server with standard name
server = FastMCP(
    "oanda-trading-optimized",
    instructions=f"""
# {'LLVM-Optimized' if NUMBA_AVAILABLE else 'Fallback'} OANDA Trading MCP Server

This server provides {'high-performance' if NUMBA_AVAILABLE else 'optimized'} tools for creating market orders using the OANDA v20 API.
{'LLVM optimizations are ACTIVE' if NUMBA_AVAILABLE else 'Running in fallback mode (install numba for LLVM acceleration)'}.

Available tools:
- create_market_order: Create a market order with {'LLVM-optimized' if NUMBA_AVAILABLE else 'optimized'} validation
- batch_validate_orders: Validate multiple orders efficiently {'using LLVM' if NUMBA_AVAILABLE else ''}

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
@performance_monitor
async def create_market_order(
    account_id: str,
    token: str, 
    instrument: str,
    units: str,
    hostname: str = "api-fxpractice.oanda.com",
    port: int = 443,
    estimated_price: float = 1.0
) -> str:
    """Create a market order using OANDA v20 API with optimizations"""
    try:
        # LLVM-optimized pre-validation
        units_float = float(units)
        is_valid_units, error_code = validate_units_llvm(units_float)
        
        if not is_valid_units:
            error_messages = {
                1: "Units cannot be zero",
                2: "Units too small (minimum 1 unit)",
                3: "Units too large (maximum 10M units)"
            }
            return json.dumps({
                "success": False,
                "error": error_messages.get(error_code, "Invalid units"),
                "status": "validation_error",
                "account_id": account_id,
                "instrument": instrument,
                "units": units,
                "llvm_enabled": NUMBA_AVAILABLE
            }, indent=2)
        
        # LLVM-optimized instrument formatting with caching
        formatted_instrument = instrument_cache.get_formatted_instrument(instrument)
        
        if formatted_instrument is None:
            return json.dumps({
                "success": False,
                "error": "Invalid instrument format. Use format like EUR_USD or EUR/USD",
                "status": "validation_error", 
                "account_id": account_id,
                "instrument": instrument,
                "units": units,
                "llvm_enabled": NUMBA_AVAILABLE
            }, indent=2)
        
        # Calculate margin requirement using LLVM
        margin_required = calculate_margin_requirement_llvm(units_float, estimated_price)
        
        # Create the API context
        api = v20.Context(hostname, port, token=token)
        
        # Submit the market order request
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
            "units": units,
            "margin_required": round(margin_required, 2),
            "llvm_enabled": NUMBA_AVAILABLE
        }
        
        # Add transaction details if available
        if hasattr(response, 'body') and response.body:
            if hasattr(response.body, 'orderCreateTransaction'):
                transaction = response.body.orderCreateTransaction
                result["orderCreateTransaction"] = transaction.dict() if hasattr(transaction, 'dict') else str(transaction)
            
            if hasattr(response.body, 'orderFillTransaction'):
                fill_transaction = response.body.orderFillTransaction
                result["orderFillTransaction"] = fill_transaction.dict() if hasattr(fill_transaction, 'dict') else str(fill_transaction)
            
            if hasattr(response.body, 'orderCancelTransaction'):
                cancel_transaction = response.body.orderCancelTransaction
                result["orderCancelTransaction"] = cancel_transaction.dict() if hasattr(cancel_transaction, 'dict') else str(cancel_transaction)
            
            if hasattr(response.body, 'lastTransactionID'):
                result["lastTransactionID"] = response.body.lastTransactionID
            
            if hasattr(response.body, 'relatedTransactionIDs'):
                result["relatedTransactionIDs"] = response.body.relatedTransactionIDs
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "status": "api_error",
            "account_id": account_id,
            "instrument": instrument,
            "units": units,
            "llvm_enabled": NUMBA_AVAILABLE
        }
        logger.error(f"Error creating market order: {e}")
        return json.dumps(error_result, indent=2)

@server.tool(
    name="batch_validate_orders",
    description="""Validate multiple orders efficiently using optimizations.

Args:
    orders: List[Dict]
        List of order dictionaries with 'units' and 'estimated_price' keys
    max_total_margin: float
        Maximum total margin allowed for all orders combined
""",
)
@performance_monitor
async def batch_validate_orders(orders: List[Dict[str, Any]], max_total_margin: float = 100000.0) -> str:
    """Validate multiple orders efficiently using optimizations"""
    try:
        if not orders:
            return json.dumps({
                "valid_orders": [], 
                "total_margin": 0.0, 
                "validation_summary": "No orders provided",
                "llvm_enabled": NUMBA_AVAILABLE
            })
        
        # Convert to NumPy arrays for LLVM processing
        units_array = np.array([float(order.get('units', 0)) for order in orders], dtype=np.float64)
        prices_array = np.array([float(order.get('estimated_price', 1.0)) for order in orders], dtype=np.float64)
        
        # LLVM-optimized batch validation
        valid_flags = batch_validate_orders_llvm(units_array, prices_array, max_total_margin)
        
        # Process results
        valid_orders = []
        total_margin = 0.0
        
        for i, (order, is_valid) in enumerate(zip(orders, valid_flags)):
            if is_valid:
                units = units_array[i]
                price = prices_array[i]
                margin = calculate_margin_requirement_llvm(units, price)
                
                order_result = {
                    **order,
                    "index": i,
                    "is_valid": True,
                    "margin_required": round(margin, 2)
                }
                valid_orders.append(order_result)
                total_margin += margin
        
        result = {
            "valid_orders": valid_orders,
            "total_orders_submitted": len(orders),
            "valid_orders_count": len(valid_orders),
            "rejection_count": len(orders) - len(valid_orders),
            "total_margin_required": round(total_margin, 2),
            "max_margin_allowed": max_total_margin,
            "margin_utilization": round((total_margin / max_total_margin) * 100, 2) if max_total_margin > 0 else 0,
            "validation_summary": f"{len(valid_orders)}/{len(orders)} orders passed validation",
            "llvm_enabled": NUMBA_AVAILABLE
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "status": "batch_validation_error",
            "llvm_enabled": NUMBA_AVAILABLE
        }
        logger.error(f"Error in batch validation: {e}")
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    print("Starting OANDA Trading MCP server...")
    if NUMBA_AVAILABLE:
        print("✓ LLVM-optimized functions ENABLED")
        print("✓ Performance optimizations active:")
        print("  - LLVM-compiled validation functions")
        print("  - Intelligent instrument caching")
        print("  - Batch processing capabilities")
        print("  - Performance monitoring")
    else:
        print("⚠ Running in fallback mode")
        print("Install Numba for LLVM acceleration: pip install numba")
        print("Current optimizations:")
        print("  - Intelligent instrument caching")
        print("  - Batch processing capabilities") 
        print("  - Performance monitoring")
    
    server.run(transport="stdio")