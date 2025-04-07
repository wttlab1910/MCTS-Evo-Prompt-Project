"""
Application entry point.
"""
import argparse
from app.api.routes import create_app
from app.config import API_CONFIG

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MCTS-Evo-Prompt Server")
    parser.add_argument(
        "--host", default=API_CONFIG["host"],
        help=f"Host address (default: {API_CONFIG['host']})"
    )
    parser.add_argument(
        "--port", type=int, default=API_CONFIG["port"],
        help=f"Port number (default: {API_CONFIG['port']})"
    )
    parser.add_argument(
        "--debug", action="store_true", default=API_CONFIG["debug"],
        help="Enable debug mode"
    )
    parser.add_argument(
        "--reload", action="store_true", default=API_CONFIG["reload"],
        help="Enable auto-reload on code changes"
    )
    parser.add_argument(
        "--workers", type=int, default=API_CONFIG["workers"],
        help=f"Number of worker processes (default: {API_CONFIG['workers']})"
    )
    
    return parser.parse_args()

def main():
    """Run the application server."""
    args = parse_args()
    
    app = create_app()
    
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()