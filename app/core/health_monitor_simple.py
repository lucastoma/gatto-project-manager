"""
Simplified Health Monitor for GattoNero AI Assistant
=====================================================

A streamlined version focusing on core health monitoring functionality.
"""

import time
import psutil
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import json

from .development_logger import get_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    # Poprawka: `details` i `timestamp` mogą być None przy inicjalizacji, więc oznaczono jako Optional
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        # Inicjalizacja wartości domyślnych, jeśli nie zostały podane
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SimpleHealthMonitor:
    """Simplified health monitoring system."""
    
    def __init__(self):
        self.logger = get_logger()
        self._results: Dict[str, HealthResult] = {}
        self._algorithm_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Simple Health Monitor initialized")
    
    def check_system_memory(self) -> HealthResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > 75:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthResult(
                status=status,
                message=message,
                details={"memory_percent": memory_percent}
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory: {str(e)}"
            )
    
    def check_disk_space(self) -> HealthResult:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)
            
            if disk_percent > 95 or free_gb < 1.0:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {disk_percent:.1f}% used"
            elif disk_percent > 85 or free_gb < 5.0:
                status = HealthStatus.WARNING
                message = f"Low disk space: {disk_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space adequate: {disk_percent:.1f}% used"
            
            return HealthResult(
                status=status,
                message=message,
                details={"disk_percent": disk_percent, "free_gb": free_gb}
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}"
            )
    
    def check_python_environment(self) -> HealthResult:
        """Check Python environment health."""
        try:
            import sys
            python_version = sys.version_info
            
            if python_version < (3, 8):
                status = HealthStatus.WARNING
                message = f"Python {python_version.major}.{python_version.minor} is outdated"
            else:
                status = HealthStatus.HEALTHY
                message = f"Python {python_version.major}.{python_version.minor} is adequate"
            
            return HealthResult(
                status=status,
                message=message,
                details={"python_version": f"{python_version.major}.{python_version.minor}"}
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to check Python environment: {str(e)}"
            )
    
    def run_all_checks(self) -> Dict[str, HealthResult]:
        """Run all health checks."""
        checks = {
            "memory": self.check_system_memory,
            "disk": self.check_disk_space,
            "python": self.check_python_environment
        }
        
        results = {}
        for name, check_func in checks.items():
            try:
                result = check_func()
                results[name] = result
                self._results[name] = result
            except Exception as e:
                error_result = HealthResult(
                    status=HealthStatus.CRITICAL,
                    message=f"Health check {name} failed: {str(e)}"
                )
                results[name] = error_result
                self._results[name] = error_result
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        # Run fresh checks
        self.run_all_checks()
        
        # Determine overall status
        critical_count = sum(1 for r in self._results.values() if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in self._results.values() if r.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.value,
            "summary": {
                "total_checks": len(self._results),
                "healthy": sum(1 for r in self._results.values() if r.status == HealthStatus.HEALTHY),
                "warnings": warning_count,
                "critical": critical_count
            },
            "checks": {
                # Poprawka: Upewnienie się, że timestamp nie jest None
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else "N/A"
                }
                for name, result in self._results.items()
            }
        }
    
    def record_algorithm_call(self, algorithm_id: str, duration_ms: float, success: bool = True):
        """Record algorithm performance data."""
        if algorithm_id not in self._algorithm_stats:
            self._algorithm_stats[algorithm_id] = {
                "total_calls": 0,
                "error_count": 0,
                "total_duration": 0.0,
                "last_call": None
            }
        
        stats = self._algorithm_stats[algorithm_id]
        stats["total_calls"] += 1
        stats["total_duration"] += duration_ms
        stats["last_call"] = datetime.now()
        
        if not success:
            stats["error_count"] += 1


# Global simple health monitor instance
_global_simple_monitor: Optional[SimpleHealthMonitor] = None

def get_simple_health_monitor() -> SimpleHealthMonitor:
    """Get or create global simple health monitor instance."""
    global _global_simple_monitor
    if _global_simple_monitor is None:
        _global_simple_monitor = SimpleHealthMonitor()
    return _global_simple_monitor


if __name__ == "__main__":
    # Demo and testing
    monitor = SimpleHealthMonitor()
    
    print("Testing Simple Health Monitor...")
    
    # Run all checks
    results = monitor.run_all_checks()
    print(f"Health check results: {len(results)} checks completed")
    
    for name, result in results.items():
        print(f"  {name}: {result.status.value} - {result.message}")
    
    # Get health status
    status = monitor.get_health_status()
    print(f"\nOverall status: {status['overall_status']}")
    print(f"Summary: {status['summary']}")
