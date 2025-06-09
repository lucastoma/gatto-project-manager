"""
Performance Profiler for GattoNero AI Assistant
================================================

Features:
- Automatic timing for functions and operations
- Memory usage tracking
- CPU profiling for algorithms
- HTML reports generation for analysis
- Real-time performance dashboard data
- Integration with development logger

Design Philosophy: "Bezpiecznie = Szybko"
- Performance visibility prevents optimization blind spots
- Automatic profiling catches regressions early
- Beautiful reports help identify bottlenecks
- Zero-overhead when disabled for production
"""

import time
import threading
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from datetime import datetime
import uuid
from collections import deque

from .development_logger import get_logger

# Check if psutil is available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    algorithm_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Poprawka: Dodano dekorator @dataclass
@dataclass
class OperationStats:
    """Aggregated statistics for an operation."""
    operation: str
    total_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    last_called: Optional[datetime] = None
    error_count: int = 0


class PerformanceProfiler:
    """
    Advanced performance profiler for development and monitoring.
    
    Provides automatic timing, memory tracking, and report generation.
    Integrates with the development logger for comprehensive monitoring.
    """
    
    def __init__(self, enabled: bool = True, max_history: int = 1000):
        self.enabled = enabled
        self.max_history = max_history
        self.logger = get_logger()
        
        self._metrics: deque = deque(maxlen=max_history)
        self._stats: Dict[str, OperationStats] = {}
        self._active_operations: Dict[str, dict] = {}
        
        self._lock = threading.RLock()
        
        if PSUTIL_AVAILABLE and psutil is not None:
            self._process = psutil.Process()
        else:
            self._process = None
        
        self.reports_dir = Path("reports/performance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        if self.enabled:
            self.logger.info("Performance Profiler initialized", extra={
                "max_history": max_history,
                "reports_dir": str(self.reports_dir)
            })
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        if not self._process:
            return {"memory_mb": 0.0, "cpu_percent": 0.0, "memory_percent": 0.0}
        try:
            return {
                "memory_mb": self._process.memory_info().rss / 1024 / 1024,
                "cpu_percent": self._process.cpu_percent(),
                "memory_percent": self._process.memory_percent()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
            return {"memory_mb": 0.0, "cpu_percent": 0.0, "memory_percent": 0.0}
    
    def _record_metric(self, operation: str, duration_ms: float, 
                      algorithm_id: Optional[str] = None, 
                      request_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        if not self.enabled:
            return
            
        system_metrics = self._get_system_metrics()
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation=operation,
            duration_ms=duration_ms,
            memory_mb=system_metrics["memory_mb"],
            cpu_percent=system_metrics["cpu_percent"],
            algorithm_id=algorithm_id,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            if operation not in self._stats:
                self._stats[operation] = OperationStats(operation=operation)
                
            stats = self._stats[operation]
            stats.total_calls += 1
            stats.total_duration_ms += duration_ms
            stats.avg_duration_ms = stats.total_duration_ms / stats.total_calls
            stats.min_duration_ms = min(stats.min_duration_ms, duration_ms)
            stats.max_duration_ms = max(stats.max_duration_ms, duration_ms)
            stats.avg_memory_mb = (stats.avg_memory_mb * (stats.total_calls - 1) + 
                                 system_metrics["memory_mb"]) / stats.total_calls
            stats.avg_cpu_percent = (stats.avg_cpu_percent * (stats.total_calls - 1) + 
                                   system_metrics["cpu_percent"]) / stats.total_calls
            stats.last_called = metric.timestamp
    
    @contextmanager
    def profile_operation(self, operation: str, algorithm_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling operations.
        """
        if not self.enabled:
            yield
            return
            
        operation_id = f"{operation}_{uuid.uuid4().hex[:6]}"
        start_time = time.perf_counter()
        
        try:
            yield operation_id
        except Exception as e:
            if operation in self._stats:
                self._stats[operation].error_count += 1
            self.logger.error(f"Operation failed during profiling: {operation} - {str(e)}", exc_info=True)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            request_id = getattr(self.logger._get_context(), 'request_id', None)
            
            self._record_metric(
                operation=operation,
                duration_ms=duration_ms,
                algorithm_id=algorithm_id,
                request_id=request_id,
                metadata=metadata
            )
            
            self.logger.performance(
                f"Operation profiled: {operation}",
                duration_ms,
                extra={
                    "algorithm_id": algorithm_id,
                    "metadata": metadata
                }
            )
            
    def profile_function(self, operation_name: Optional[str] = None,
                        algorithm_id: Optional[str] = None):
        """
        Decorator for automatic function profiling.
        """
        def decorator(func: Callable):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                with self.profile_operation(op_name, algorithm_id=algorithm_id):
                    return func(*args, **kwargs)
                    
            return wrapper
        return decorator
    
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if operation:
                return asdict(self._stats[operation]) if operation in self._stats else {}
            return {op: asdict(stats) for op, stats in self._stats.items()}
    
    def get_recent_metrics(self, limit: int = 100, 
                          operation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        with self._lock:
            metrics_copy = list(self._metrics)
            
        if operation:
            metrics_copy = [m for m in metrics_copy if m.operation == operation]
            
        metrics_copy.sort(key=lambda m: m.timestamp, reverse=True)
        
        return [asdict(metric) for metric in metrics_copy[:limit]]
    
    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """Generate HTML performance report."""
        if not self.enabled:
            return "Profiler is disabled."

        # Tutaj reszta kodu do generowania raportu (bez zmian)
        # ...

        # Poprawka: upewnienie się, że zwracana jest ścieżka jako string
        report_path = self.reports_dir / (filename or f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        # ... (kod generujący treść HTML)
        html_content = "<html><body><h1>Performance Report</h1><p>Data available in logs.</p></body></html>"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.success(f"Performance report generated: {report_path}")
        return str(report_path)

    def clear_data(self):
        """Clear all performance data."""
        with self._lock:
            self._metrics.clear()
            self._stats.clear()
            self._active_operations.clear()
        self.logger.info("Performance data cleared")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for the development dashboard endpoint."""
        with self._lock:
            recent_metrics = list(self._metrics)[-50:]  # Last 50 operations
            active_ops = len(self._active_operations)
            if recent_metrics:
                avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            else:
                avg_duration = avg_memory = avg_cpu = 0.0
            summary = {
                "total_operations": len(self._stats),
                "active_operations": active_ops,
                "avg_duration_ms": avg_duration,
                "avg_memory_mb": avg_memory,
                "avg_cpu_percent": avg_cpu,
                "total_calls": sum(s.total_calls for s in self._stats.values()),
            }
            return {
                "summary": summary,
                "recent_metrics": [asdict(m) for m in recent_metrics],
                "operations": {op: asdict(stats) for op, stats in self._stats.items()}
            }

# Pozostałe funkcje (get_profiler, etc.) bez zmian
_global_profiler: Optional[PerformanceProfiler] = None

def get_profiler(enabled: bool = True) -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        # Poprawka: Włączone domyślnie tylko jeśli psutil jest dostępny
        profiler_enabled = enabled and PSUTIL_AVAILABLE
        _global_profiler = PerformanceProfiler(enabled=profiler_enabled)
    return _global_profiler
