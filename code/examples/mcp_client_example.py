#!/usr/bin/env python3
"""
Example: Using the Robust MCP Client

This demonstrates how to use the robust MCP client to interact with the MCP server
with proper error handling, timeouts, and message ID tracking.
"""
from mcp.mcp_client import create_client, RobustMCPClient


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create client with default settings
    client = create_client(debug=True)
    
    try:
        # Start client (starts server process)
        client.start()
        print("✓ Client started")
        
        # List available tools
        tools = client.list_tools()
        print(f"✓ Found {len(tools)} tools")
        
        # Call a simple tool
        result = client.call_tool("status", {})
        print(f"✓ Status check: {result.get('status', {}).get('summary', 'OK')}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.stop()
        print("✓ Client stopped\n")


def example_context_manager():
    """Using context manager for automatic cleanup."""
    print("=" * 60)
    print("Example 2: Context Manager")
    print("=" * 60)
    
    with create_client(debug=False) as client:
        # Client automatically starts
        tools = client.list_tools()
        print(f"✓ Found {len(tools)} tools")
        
        # Call multiple tools
        for tool_name in ["gpu_info", "system_software"]:
            try:
                result = client.call_tool(tool_name, {})
                print(f"✓ {tool_name}: OK")
            except Exception as e:
                print(f"✗ {tool_name}: {e}")
    
    # Client automatically stops
    print("✓ Context manager cleaned up\n")


def example_error_handling():
    """Demonstrating error handling."""
    print("=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)
    
    client = create_client(debug=True)
    
    try:
        client.start()
        
        # Try calling non-existent tool
        try:
            result = client.call_tool("nonexistent_tool", {})
        except RuntimeError as e:
            print(f"✓ Caught expected error: {e}")
        
        # Try calling with invalid arguments
        try:
            result = client.call_tool("status", {"invalid": "param"})
            print(f"✓ Tool handled invalid params gracefully")
        except Exception as e:
            print(f"✓ Caught error: {e}")
        
    finally:
        client.stop()
        print("✓ Error handling example complete\n")


def example_timeout_handling():
    """Demonstrating timeout handling."""
    print("=" * 60)
    print("Example 4: Timeout Handling")
    print("=" * 60)
    
    client = create_client(timeout=1.0, debug=True)  # 1 second timeout
    
    try:
        client.start()
        
        # Call a tool that might take longer
        try:
            # This will timeout if it takes > 1 second
            result = client.call_tool("status", {}, timeout=1.0)
            print(f"✓ Tool completed within timeout")
        except TimeoutError as e:
            print(f"✓ Caught timeout: {e}")
        
    finally:
        client.stop()
        print("✓ Timeout handling example complete\n")


def example_concurrent_requests():
    """Demonstrating concurrent request handling."""
    print("=" * 60)
    print("Example 5: Concurrent Requests")
    print("=" * 60)
    
    import threading
    
    client = create_client(debug=False)
    results = []
    errors = []
    
    def call_tool(tool_name):
        try:
            result = client.call_tool(tool_name, {})
            results.append((tool_name, "OK"))
        except Exception as e:
            errors.append((tool_name, str(e)))
    
    try:
        client.start()
        
        # Make concurrent requests
        threads = []
        tool_names = ["status", "gpu_info", "system_software"]
        
        for tool_name in tool_names:
            t = threading.Thread(target=call_tool, args=(tool_name,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        print(f"✓ Completed {len(results)} requests successfully")
        if errors:
            print(f"✗ {len(errors)} requests failed")
            for tool, error in errors:
                print(f"  - {tool}: {error}")
        
    finally:
        client.stop()
        print("✓ Concurrent requests example complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Robust MCP Client Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_context_manager()
    example_error_handling()
    example_timeout_handling()
    example_concurrent_requests()
    
    print("=" * 60)
    print("All examples complete!")
    print("=" * 60)
