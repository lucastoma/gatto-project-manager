"""WebView Package

Web interface for testing and debugging algorithms before JSX integration.

This package provides:
- Web-based algorithm testing interface
- File upload and parameter configuration
- Live result preview and debugging
- Export functionality for results

Modules:
- routes: Flask routes and API endpoints
- utils: Helper functions and utilities
- tests: Test suite for WebView functionality

Usage:
    from app.webview import webview_bp
    app.register_blueprint(webview_bp)

Version: 1.0.0
Author: GattoNero Development Team
Status: Development - Phase 1 (Basic Functionality)
"""

from .routes import webview_bp

__version__ = '1.0.0'
__author__ = 'GattoNero Development Team'
__status__ = 'Development'

# Export the blueprint for easy import
__all__ = ['webview_bp']