"""
Health Monitor for GattoNero AI Assistant
==========================================

Features:
- Algorithm health checks and status tracking
- Dependency verification (libraries, files, resources)
- System resource monitoring (memory, disk, CPU)
- Health endpoints for monitoring
- Automatic recovery suggestions
- Alert system for critical issues

Design Philosophy: "Bezpiecznie = Szybko"
- Proactive health monitoring prevents runtime failures
- Clear health status helps debug issues quickly
- Automatic checks catch problems before users hit them
- Recovery suggestions guide quick fixes
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field, asdict
import json
import importlib
import sys
import os
import subprocess
from collections import defaultdict, deque

from .development_logger import get_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Definition of a health check."""
    name: str
    check_function: Callable[[], 'HealthResult']
    interval_seconds: int = 60
    timeout_seconds: int = 10
    critical: bool = False
    description: str = ""
    category: str = "general"


@dataclass
class HealthResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlgorithmHealth:
    """Health information for an algorithm."""
    algorithm_id: str
    status: HealthStatus
    last_check: datetime
    dependencies_ok: bool
    resource_usage: Dict[str, float]
    error_count: int
    success_rate: float
    issues: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Comprehensive health monitoring system for GattoNero AI Assistant.
    
    Monitors algorithms, system resources, dependencies, and provides
    health endpoints for external monitoring.
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.logger = get_logger()
        
        # Health checks registry
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, HealthResult] = {}
        self._algorithm_health: Dict[str, AlgorithmHealth] = {}
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # System monitoring
        self._process = psutil.Process()
        self._last_check_times: Dict[str, datetime] = {}
        
        # Performance tracking for algorithms
        self._algorithm_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "error_count": 0,
            "total_duration": 0.0,
            "last_call": None,
            "recent_errors": deque(maxlen=10)
        })
        
        # Register default health checks
        self._register_default_checks()
        
        self.logger.info("Health Monitor initialized", extra={
            "check_interval": check_interval,
            "default_checks": len(self._checks)
        })
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # System resource checks
        self.register_check("system_memory", self._check_memory, 30, 
                          critical=True, description="System memory usage",
                          category="system")
        
        self.register_check("system_disk", self._check_disk_space, 60,
                          critical=True, description="Disk space availability",
                          category="system")
        
        self.register_check("system_cpu", self._check_cpu_usage, 30,
                          critical=False, description="CPU usage monitoring",
                          category="system")
        
        # Python environment checks
        self.register_check("python_environment", self._check_python_env, 300,
                          critical=True, description="Python environment health",
                          category="environment")
        
        # Flask application checks
        self.register_check("flask_app", self._check_flask_health, 60,
                          critical=True, description="Flask application health",
                          category="application")
        
        # File system checks
        self.register_check("filesystem", self._check_filesystem, 120,
                          critical=True, description="File system permissions and access",
                          category="filesystem")
    
    def register_check(self, name: str, check_function: Callable[[], HealthResult],
                      interval_seconds: int = 60, timeout_seconds: int = 10,
                      critical: bool = False, description: str = "",
                      category: str = "general"):
        """Register a new health check."""
        check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description,
            category=category
        )
        
        with self._lock:
            self._checks[name] = check
            
        self.logger.debug(f"Health check registered: {name}", extra={
            "category": category,
            "critical": critical,
            "interval": interval_seconds
        })
    
    def register_algorithm(self, algorithm_id: str, dependencies: Optional[List[str]] = None):
        """Register an algorithm for health monitoring."""
        with self._lock:
            self._algorithm_health[algorithm_id] = AlgorithmHealth(
                algorithm_id=algorithm_id,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                dependencies_ok=True,
                resource_usage={},
                error_count=0,
                success_rate=1.0
            )
        
        # Register algorithm-specific checks
        if dependencies is None:
            dependencies = []
        if dependencies:
            self.register_check(
                f"algorithm_{algorithm_id}_dependencies",
                lambda: self._check_algorithm_dependencies(algorithm_id, dependencies),
                300,  # Check every 5 minutes
                critical=True,
                description=f"Dependencies for {algorithm_id}",
                category="algorithm"
            )
        
        self.logger.info(f"Algorithm registered for health monitoring: {algorithm_id}")
    
    def record_algorithm_call(self, algorithm_id: str, duration_ms: float, 
                            success: bool = True, error: Optional[str] = None):
        """Record algorithm performance and health data."""
        with self._lock:
            stats = self._algorithm_stats[algorithm_id]
            stats["total_calls"] += 1
            stats["total_duration"] += duration_ms
            stats["last_call"] = datetime.now()
            
            if not success:
                stats["error_count"] += 1
                if error is not None:
                    stats["recent_errors"].append({
                        "timestamp": datetime.now(),
                        "error": error
                    })
            
            # Update algorithm health
            if algorithm_id in self._algorithm_health:
                health = self._algorithm_health[algorithm_id]
                health.error_count = stats["error_count"]
                health.success_rate = 1.0 - (stats["error_count"] / stats["total_calls"])
                
                # Determine health status based on recent performance
                if health.success_rate < 0.5:
                    health.status = HealthStatus.CRITICAL
                elif health.success_rate < 0.8:
                    health.status = HealthStatus.WARNING
                else:
                    health.status = HealthStatus.HEALTHY
                
                health.last_check = datetime.now()
    
    def _check_memory(self) -> HealthResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
                suggestions = [
                    "Free up memory by closing unnecessary applications",
                    "Restart the application to clear memory leaks",
                    "Consider increasing available RAM"
                ]
            elif memory_percent > 75:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
                suggestions = ["Monitor memory usage closely", "Consider optimizing algorithms"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
                suggestions = []
            
            return HealthResult(
                status=status,
                message=message,
                details={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "total_gb": memory.total / (1024**3)
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory: {str(e)}",
                suggestions=["Check system monitoring tools", "Restart monitoring service"]
            )
    
    def _check_disk_space(self) -> HealthResult:
        """Check disk space usage."""
        try:
            # Check current directory disk space
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)
            
            if disk_percent > 95 or free_gb < 1.0:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {disk_percent:.1f}% used, {free_gb:.1f}GB free"
                suggestions = [
                    "Clean up temporary files",
                    "Remove old log files",
                    "Archive or delete unnecessary files"
                ]
            elif disk_percent > 85 or free_gb < 5.0:
                status = HealthStatus.WARNING
                message = f"Low disk space: {disk_percent:.1f}% used, {free_gb:.1f}GB free"
                suggestions = ["Monitor disk usage", "Plan for disk cleanup"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space adequate: {disk_percent:.1f}% used, {free_gb:.1f}GB free"
                suggestions = []
            
            return HealthResult(
                status=status,
                message=message,
                details={
                    "disk_percent": disk_percent,
                    "free_gb": free_gb,
                    "used_gb": disk_usage.used / (1024**3),
                    "total_gb": disk_usage.total / (1024**3)
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                suggestions=["Check disk access permissions", "Verify disk health"]
            )
    
    def _check_cpu_usage(self) -> HealthResult:
        """Check CPU usage."""
        try:
            cpu_percent = self._process.cpu_percent(interval=1)
            
            if cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
                suggestions = ["Monitor for CPU-intensive operations", "Consider algorithm optimization"]
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
                suggestions = []
            
            # os.getloadavg is not available on Windows
            load_average = None
            if hasattr(os, 'getloadavg') and callable(getattr(os, 'getloadavg', None)):
                try:
                    load_average = os.getloadavg()  # type: ignore[attr-defined]
                except Exception:
                    load_average = None
            
            return HealthResult(
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "load_average": load_average
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.WARNING,
                message=f"Failed to check CPU usage: {str(e)}",
                suggestions=["Check system monitoring availability"]
            )
    
    def _check_python_env(self) -> HealthResult:
        """Check Python environment health."""
        try:
            issues = []
            suggestions = []
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                issues.append(f"Python version {python_version.major}.{python_version.minor} is outdated")
                suggestions.append("Upgrade to Python 3.8 or higher")
            
            # Check critical modules
            critical_modules = ['flask', 'numpy', 'PIL', 'psutil']
            missing_modules = []
            
            for module in critical_modules:
                try:
                    importlib.import_module(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                issues.append(f"Missing critical modules: {', '.join(missing_modules)}")
                suggestions.append("Install missing modules with pip")
            
            # Determine status
            if missing_modules or python_version < (3, 7):
                status = HealthStatus.CRITICAL
            elif issues:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            message = "Python environment healthy" if not issues else f"Issues found: {'; '.join(issues)}"
            
            return HealthResult(
                status=status,
                message=message,
                details={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "missing_modules": missing_modules,
                    "executable": sys.executable
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check Python environment: {str(e)}",
                suggestions=["Check Python installation", "Verify module accessibility"]
            )
    
    def _check_flask_health(self) -> HealthResult:
        """Check Flask application health."""
        try:
            # This is a basic check - in a real setup you might check routes, database connections, etc.
            from flask import current_app
            
            # Check if Flask app is running
            if current_app:
                status = HealthStatus.HEALTHY
                message = "Flask application running"
                details = {
                    "app_name": current_app.name,
                    "debug_mode": current_app.debug,
                    "testing": current_app.testing
                }
            else:
                status = HealthStatus.WARNING
                message = "Flask application context not available"
                details = {}
            
            return HealthResult(
                status=status,
                message=message,
                details=details,
                suggestions=[] if status == HealthStatus.HEALTHY else ["Check Flask application startup"]
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.WARNING,
                message=f"Flask health check failed: {str(e)}",
                suggestions=["Check Flask application configuration", "Verify application startup"]
            )
    
    def _check_filesystem(self) -> HealthResult:
        """Check filesystem health and permissions."""
        try:
            issues = []
            suggestions = []
            
            # Check critical directories
            critical_dirs = ['app', 'logs', 'uploads', 'results']
            
            for dir_name in critical_dirs:
                dir_path = Path(dir_name)
                
                if not dir_path.exists():
                    issues.append(f"Directory {dir_name} does not exist")
                    suggestions.append(f"Create directory: {dir_name}")
                elif not os.access(dir_path, os.R_OK | os.W_OK):
                    issues.append(f"Insufficient permissions for {dir_name}")
                    suggestions.append(f"Fix permissions for {dir_name}")
            
            # Check temp directory writability
            try:
                temp_file = Path("temp_health_check.txt")
                temp_file.write_text("health check")
                temp_file.unlink()
            except Exception:
                issues.append("Cannot write to current directory")
                suggestions.append("Check directory write permissions")
            
            status = HealthStatus.CRITICAL if issues else HealthStatus.HEALTHY
            message = "Filesystem healthy" if not issues else f"Filesystem issues: {'; '.join(issues)}"
            
            return HealthResult(
                status=status,
                message=message,
                details={"issues": issues, "checked_directories": critical_dirs},
                suggestions=suggestions
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Filesystem check failed: {str(e)}",
                suggestions=["Check filesystem access", "Verify directory permissions"]
            )
    
    def _check_algorithm_dependencies(self, algorithm_id: str, dependencies: List[str]) -> HealthResult:
        """Check algorithm dependencies."""
        try:
            missing_deps = []
            
            for dep in dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                status = HealthStatus.CRITICAL
                message = f"Algorithm {algorithm_id} missing dependencies: {', '.join(missing_deps)}"
                suggestions = [f"Install missing dependencies: {', '.join(missing_deps)}"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Algorithm {algorithm_id} dependencies satisfied"
                suggestions = []
            
            return HealthResult(
                status=status,
                message=message,
                details={
                    "algorithm_id": algorithm_id,
                    "dependencies": dependencies,
                    "missing": missing_deps
                },
                suggestions=suggestions
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check dependencies for {algorithm_id}: {str(e)}",
                suggestions=["Check dependency configuration", "Verify import paths"]
            )
    
    def run_check(self, check_name: str) -> Optional[HealthResult]:
        """Run a specific health check."""
        if check_name not in self._checks:
            self.logger.warning(f"Unknown health check: {check_name}")
            return None
        
        check = self._checks[check_name]
        
        try:
            start_time = time.time()
            result = check.check_function()
            duration = time.time() - start_time
            
            with self._lock:
                self._results[check_name] = result
                self._last_check_times[check_name] = datetime.now()
            
            self.logger.debug(f"Health check completed: {check_name}", extra={
                "status": result.status.value,
                "duration_ms": duration * 1000,
                "check_message": result.message  # Renamed to avoid conflict
            })
            
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.logger.warning(f"Health issue detected in {check_name}: {result.message}")
            
            return result
            
        except Exception as e:
            error_result = HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Health check {check_name} failed: {str(e)}",
                suggestions=["Check health check implementation", "Review system logs"]
            )
            
            with self._lock:
                self._results[check_name] = error_result
                
            self.logger.error(f"Health check failed: {check_name} - {str(e)}")
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthResult]:
        """Run all registered health checks."""
        results = {}
        
        for check_name in self._checks:
            result = self.run_check(check_name)
            if result:
                results[check_name] = result
                
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self._lock:
            # Overall status determination
            critical_issues = []
            warning_issues = []
            
            for check_name, result in self._results.items():
                if result.status == HealthStatus.CRITICAL:
                    critical_issues.append(check_name)
                elif result.status == HealthStatus.WARNING:
                    warning_issues.append(check_name)
            
            if critical_issues:
                overall_status = HealthStatus.CRITICAL
            elif warning_issues:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status.value,
                "summary": {
                    "total_checks": len(self._checks),
                    "healthy": len([r for r in self._results.values() if r.status == HealthStatus.HEALTHY]),
                    "warnings": len(warning_issues),
                    "critical": len(critical_issues)
                },
                "critical_issues": critical_issues,
                "warning_issues": warning_issues,
                "checks": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "timestamp": result.timestamp.isoformat(),
                        "suggestions": result.suggestions
                    }
                    for name, result in self._results.items()
                },
                "algorithms": {
                    alg_id: {
                        "status": health.status.value,
                        "success_rate": health.success_rate,
                        "error_count": health.error_count,
                        "last_check": health.last_check.isoformat()
                    }
                    for alg_id, health in self._algorithm_health.items()
                }
            }
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Health monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
            
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                current_time = datetime.now()
                
                # Check which health checks need to run
                for check_name, check in self._checks.items():
                    last_check = self._last_check_times.get(check_name)
                    
                    if (last_check is None or 
                        current_time - last_check >= timedelta(seconds=check.interval_seconds)):
                        self.run_check(check_name)
                
                # Sleep until next check cycle
                self._stop_monitoring.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}")
                self._stop_monitoring.wait(5)  # Wait 5 seconds before retrying


# Global health monitor instance
_global_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor()
    return _global_monitor


if __name__ == "__main__":
    # Demo and testing
    monitor = HealthMonitor(check_interval=10)
    
    print("Testing Health Monitor...")
    
    # Register a test algorithm
    monitor.register_algorithm("test_algorithm", ["numpy", "PIL"])
    
    # Run all checks
    results = monitor.run_all_checks()
    print(f"\nInitial health check results: {len(results)} checks completed")
    
    # Get health status
    status = monitor.get_health_status()
    print(f"Overall status: {status['overall_status']}")
    print(f"Summary: {status['summary']}")
    
    # Record some algorithm calls
    monitor.record_algorithm_call("test_algorithm", 150.0, success=True)
    monitor.record_algorithm_call("test_algorithm", 75.0, success=True)
    monitor.record_algorithm_call("test_algorithm", 200.0, success=False, error="Test error")
    
    # Start monitoring
    monitor.start_monitoring()
    print("\nHealth monitoring started...")
    
    # Let it run for a bit
    import time
    time.sleep(5)
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("Health monitoring stopped")
    
    # Final status
    final_status = monitor.get_health_status()
    print(f"\nFinal overall status: {final_status['overall_status']}")
