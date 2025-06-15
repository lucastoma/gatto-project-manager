"""
Enhanced Development Logger for GattoNero AI Assistant
=======================================================

Features:
- Structured logging with JSON output for parsing
- Beautiful console output with colors for development  
- File logging with rotation for persistence
- Context tracking (request_id, operation_id)
- Performance timing integration ready
- Multiple output levels and filtering

Design Philosophy: "Bezpiecznie = Szybko"
- Clear visibility into what's happening
- Easy debugging with context
- Performance insights built-in
- Development-friendly formatting
"""

import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from typing import Optional, Dict, Any
import uuid
import time
import threading
from dataclasses import dataclass, asdict


# ANSI Color codes for beautiful console output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Semantic colors
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    DEBUG = CYAN
    SUCCESS = GREEN
    PERFORMANCE = MAGENTA


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    algorithm_id: Optional[str] = None
    user_session: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None


class DevelopmentFormatter(logging.Formatter):
    """Custom formatter for beautiful development console output."""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        # Get timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Level with color
        level_colors = {
            'DEBUG': Colors.DEBUG,
            'INFO': Colors.INFO,
            'WARNING': Colors.WARNING,
            'ERROR': Colors.ERROR,
            'CRITICAL': Colors.ERROR + Colors.BOLD
        }
        level_color = level_colors.get(record.levelname, Colors.WHITE)
        level_str = f"{level_color}{record.levelname:8}{Colors.END}"
        
        # Module/function context
        module_info = f"{Colors.CYAN}{record.name}{Colors.END}"
        if hasattr(record, 'funcName') and record.funcName:
            module_info += f".{Colors.CYAN}{record.funcName}{Colors.END}"
            
        # Context information
        context_parts = []
        if getattr(record, 'request_id', None):
            context_parts.append(f"req:{getattr(record, 'request_id')[:8]}")
        if getattr(record, 'operation_id', None):
            context_parts.append(f"op:{getattr(record, 'operation_id')[:8]}")
        if getattr(record, 'algorithm_id', None):
            context_parts.append(f"alg:{getattr(record, 'algorithm_id')}")
            
        context_str = ""
        if context_parts:
            context_str = f" {Colors.YELLOW}[{' '.join(context_parts)}]{Colors.END}"
            
        # Performance information
        perf_str = ""
        duration_ms = getattr(record, 'duration_ms', None)
        if duration_ms is not None:
            if duration_ms < 10:
                perf_color = Colors.SUCCESS
            elif duration_ms < 100:
                perf_color = Colors.WARNING
            else:
                perf_color = Colors.ERROR
            perf_str = f" {perf_color}({duration_ms:.1f}ms){Colors.END}"
            
        # Main message
        message = record.getMessage()
        
        # Assemble final message
        return f"{Colors.WHITE}{timestamp}{Colors.END} {level_str} {module_info}{context_str} {message}{perf_str}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging to files."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': getattr(record, 'funcName', None),
            'line': record.lineno,
        }
        
        # Add context information safely
        context_fields = ['request_id', 'operation_id', 'algorithm_id', 'user_session']
        for field in context_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
                
        # Add performance data safely
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = getattr(record, 'duration_ms')
        if hasattr(record, 'performance_data'):
            log_data['performance_data'] = getattr(record, 'performance_data')
            
        # Add exception information
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class DevelopmentLogger:
    """
    Enhanced development logger for GattoNero AI Assistant.
    
    Provides both beautiful console output and structured JSON file logging.
    Includes context tracking and performance integration.
    """
    
    def __init__(self, name: str = "gattonero", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thread-local storage for context
        self._local = threading.local()
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Setup console handler with beautiful formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(DevelopmentFormatter())
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Setup file handler with JSON formatting
        log_file = self.log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # Setup error file handler
        error_file = self.log_dir / f"{name}_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setFormatter(JSONFormatter())
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        self.logger.info("Development Logger initialized", extra=self._get_extra())
        
    def _get_context(self) -> LogContext:
        """Get current thread-local context."""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        return self._local.context
        
    def _get_extra(self) -> Dict[str, Any]:
        """Get extra fields for logging from current context."""
        context = self._get_context()
        return asdict(context)
        
    def set_request_context(self, request_id: Optional[str] = None):
        """Set request context for current thread."""
        context = self._get_context()
        context.request_id = request_id or str(uuid.uuid4())[:8]
        
    def set_operation_context(self, operation_id: str):
        """Set operation context for current thread."""
        context = self._get_context()
        context.operation_id = operation_id
        
    def set_algorithm_context(self, algorithm_id: str):
        """Set algorithm context for current thread."""
        context = self._get_context()
        context.algorithm_id = algorithm_id
        
    def clear_context(self):
        """Clear all context for current thread."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
            
    @contextmanager
    def operation(self, operation_name: str, algorithm_id: Optional[str] = None):
        """
        Context manager for tracking operations with automatic timing.
        
        Usage:
            with logger.operation("palette_analysis", "algorithm_01_palette"):
                # Your operation code here
                pass
        """
        operation_id = f"{operation_name}_{uuid.uuid4().hex[:6]}"
        old_operation_id = getattr(self._get_context(), 'operation_id', None)
        old_algorithm_id = getattr(self._get_context(), 'algorithm_id', None)
        
        # Set new context
        self.set_operation_context(operation_id)
        if algorithm_id:
            self.set_algorithm_context(algorithm_id)
            
        start_time = time.time()
        
        try:
            self.info(f"Started operation: {operation_name}")
            yield operation_id
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            extra = self._get_extra()
            extra['duration_ms'] = duration_ms
            self.error(f"Operation failed: {operation_name} - {str(e)}", extra=extra, exc_info=True)
            raise
            
        else:
            duration_ms = (time.time() - start_time) * 1000
            extra = self._get_extra()
            extra['duration_ms'] = duration_ms
            self.info(f"Completed operation: {operation_name}", extra=extra)
            
        finally:
            # Restore previous context
            context = self._get_context()
            context.operation_id = old_operation_id
            context.algorithm_id = old_algorithm_id
            
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra={**self._get_extra(), **(extra or {}), **kwargs})
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message."""
        self.logger.info(message, extra={**self._get_extra(), **(extra or {}), **kwargs})
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={**self._get_extra(), **(extra or {}), **kwargs})
        
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message."""
        # Separate exc_info from other kwargs to avoid conflicts
        exc_info = kwargs.pop('exc_info', None)
        self.logger.error(message, extra={**self._get_extra(), **(extra or {}), **kwargs}, exc_info=exc_info)
        
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra={**self._get_extra(), **(extra or {}), **kwargs})
        
    def success(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log success message (info level with success context)."""
        success_extra = {**self._get_extra(), **(extra or {}), 'type': 'success', **kwargs}
        self.logger.info(message, extra=success_extra)
        
    def performance(self, message: str, duration_ms: float, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log performance information."""
        perf_extra = {**self._get_extra(), **(extra or {}), 'duration_ms': duration_ms, 'type': 'performance', **kwargs}
        self.logger.info(message, extra=perf_extra)


# Global logger instance
_global_logger: Optional[DevelopmentLogger] = None

def get_logger(name: str = "gattonero") -> DevelopmentLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DevelopmentLogger(name)
    return _global_logger


def setup_flask_logging(app, logger: Optional[DevelopmentLogger] = None):
    """Setup Flask request logging integration."""
    if logger is None:
        logger = get_logger()
        
    @app.before_request
    def before_request():
        from flask import request
        logger.set_request_context()
        logger.debug(f"Request started: {request.method} {request.path}")
        
    @app.after_request
    def after_request(response):
        logger.debug(f"Request completed: {response.status_code}")
        return response
        
    @app.teardown_request
    def teardown_request(exception):
        if exception:
            logger.error(f"Request error: {str(exception)}", exc_info=True)
        logger.clear_context()
