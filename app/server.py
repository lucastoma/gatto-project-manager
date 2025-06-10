"""
Enhanced Flask Server for GattoNero AI Assistant
================================================

Enhanced infrastructure features:
- Structured development logging with beautiful console output
- Performance profiling with HTML reports
- Health monitoring for algorithms and system resources
- Development dashboard endpoints
- Async processing support (future)

Design Philosophy: "Bezpiecznie = Szybko"
- Comprehensive monitoring prevents surprises
- Beautiful development experience improves productivity
- Performance insights guide optimization
- Health checks catch issues early
"""

import os
import threading
from pathlib import Path
from flask import Flask, jsonify, request

# Import enhanced infrastructure
from .core.development_logger import get_logger, setup_flask_logging
from .core.performance_profiler import get_profiler
from .core.health_monitor_simple import get_simple_health_monitor

# Import existing API routes
from .api.routes import app as api_blueprint

# Import WebView routes
from .webview import webview_bp

# Initialize enhanced infrastructure
logger = get_logger("gattonero_server")
profiler = get_profiler(enabled=True)
health_monitor = get_simple_health_monitor()

# Create enhanced Flask app
app = Flask(__name__)

# Setup development logging for Flask
setup_flask_logging(app, logger)

# Register existing API routes Blueprint
app.register_blueprint(api_blueprint)

# Register WebView Blueprint
app.register_blueprint(webview_bp)

# Debug endpoint to list all routes
@app.route('/routes')
def list_routes():
    """List all registered routes for debugging."""
    import urllib.parse
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods or set())
        output.append(f"{rule.rule} [{methods}]")
    return "<br>".join(sorted(output))

# Simple root endpoint
@app.route('/')
def root():
    """Root endpoint."""
    return jsonify({
        "status": "ok",
        "message": "GattoNero AI Assistant Server",
        "version": "Enhanced Infrastructure",
        "endpoints": {
            "health": "/api/health",
            "performance": "/api/performance/dashboard",
            "routes": "/routes"
        }
    })

# Enhanced infrastructure endpoints
@app.route('/api/health')
def health_endpoint():
    """Health check endpoint for monitoring."""
    with profiler.profile_operation("health_check"):
        health_status = health_monitor.get_health_status()
        
    return jsonify({
        "status": "ok",
        "health": health_status
    })

@app.route('/api/health/quick')
def health_quick_endpoint():
    """Quick health check for load balancers."""
    return jsonify({
        "status": "ok",
        "timestamp": health_monitor.get_health_status()["timestamp"]
    })

@app.route('/api/performance/dashboard')
def performance_dashboard():
    """Performance dashboard data endpoint."""
    with profiler.profile_operation("performance_dashboard"):
        dashboard_data = profiler.get_dashboard_data()
        
    return jsonify(dashboard_data)

@app.route('/api/performance/report')
def performance_report():
    """Generate and return performance report."""
    with profiler.profile_operation("generate_performance_report"):
        report_path = profiler.generate_html_report()
        
    return jsonify({
        "status": "success",
        "report_path": report_path,
        "message": "Performance report generated"
    })

@app.route('/api/performance/stats')
def performance_stats():
    """Get performance statistics."""
    operation = request.args.get('operation')
    stats = profiler.get_statistics(operation)
    
    return jsonify({
        "status": "success",
        "statistics": stats
    })

@app.route('/api/system/info')
def system_info():
    """System information endpoint."""
    import psutil
    import sys
    
    return jsonify({
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "flask_debug": app.debug,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.Process().cpu_percent(),
        "algorithms_registered": len(health_monitor._algorithm_stats),
        "performance_metrics": len(profiler._metrics)
    })

@app.route('/api/logs/recent')
def recent_logs():
    """Get recent log entries (if available)."""
    # This would need log file parsing in a real implementation
    return jsonify({
        "status": "info",
        "message": "Recent logs endpoint - implementation needed",
        "logs": []
    })

@app.route('/development/dashboard')
def development_dashboard():
    """Development dashboard HTML page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GattoNero Development Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-healthy { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .status-critical { color: #e74c3c; }
            .metric { display: inline-block; margin: 10px 20px; text-align: center; }
            .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
            .metric-label { color: #7f8c8d; }
            button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
            button:hover { background: #2980b9; }
            pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
        </style>
        <script>
            async function loadData() {
                try {
                    const [healthData, perfData, sysData] = await Promise.all([
                        fetch('/api/health').then(r => r.json()),
                        fetch('/api/performance/dashboard').then(r => r.json()),
                        fetch('/api/system/info').then(r => r.json())
                    ]);
                    
                    updateDashboard(healthData, perfData, sysData);
                } catch (error) {
                    console.error('Failed to load dashboard data:', error);
                }
            }
            
            function updateDashboard(health, perf, sys) {
                // Update health status
                const healthEl = document.getElementById('health-status');
                healthEl.className = `status-${health.health.overall_status}`;
                healthEl.textContent = health.health.overall_status.toUpperCase();
                
                // Update metrics
                document.getElementById('total-ops').textContent = perf.summary.total_operations;
                document.getElementById('active-ops').textContent = perf.summary.active_operations;
                document.getElementById('avg-duration').textContent = perf.summary.avg_duration_ms.toFixed(1) + 'ms';
                document.getElementById('memory-usage').textContent = sys.memory_usage_mb.toFixed(1) + 'MB';
                
                // Update details
                document.getElementById('health-details').textContent = JSON.stringify(health.health.summary, null, 2);
                document.getElementById('perf-details').textContent = JSON.stringify(perf.summary, null, 2);
            }
            
            async function generateReport() {
                try {
                    const response = await fetch('/api/performance/report');
                    const data = await response.json();
                    alert('Report generated: ' + data.report_path);
                } catch (error) {
                    alert('Failed to generate report: ' + error.message);
                }
            }
            
            // Auto-refresh every 5 seconds
            setInterval(loadData, 5000);
            
            // Load initial data
            window.onload = loadData;
        </script>
    </head>
    <body>
        <h1>🚀 GattoNero Development Dashboard</h1>
        
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="metric">
                <div class="metric-value" id="health-status">LOADING</div>
                <div class="metric-label">Health Status</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="total-ops">-</div>
                <div class="metric-label">Total Operations</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="active-ops">-</div>
                <div class="metric-label">Active Operations</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="avg-duration">-</div>
                <div class="metric-label">Avg Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="memory-usage">-</div>
                <div class="metric-label">Memory Usage</div>
            </div>
        </div>
        
        <div class="card">
            <h2>🔧 Actions</h2>
            <button onclick="loadData()">Refresh Data</button>
            <button onclick="generateReport()">Generate Performance Report</button>
            <button onclick="window.open('/api/health', '_blank')">View Health Details</button>
            <button onclick="window.open('/api/performance/dashboard', '_blank')">View Performance Data</button>
        </div>
        
        <div class="card">
            <h2>❤️ Health Details</h2>
            <pre id="health-details">Loading...</pre>
        </div>
        
        <div class="card">
            <h2>⚡ Performance Details</h2>
            <pre id="perf-details">Loading...</pre>
        </div>
        
        <div class="card">
            <h2>📚 Quick Links</h2>
            <ul>
                <li><a href="/api/health">Health Status API</a></li>
                <li><a href="/api/performance/dashboard">Performance Dashboard API</a></li>
                <li><a href="/api/system/info">System Information</a></li>
                <li><a href="/api/performance/stats">Performance Statistics</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

def initialize_server():
    """Initialize the enhanced server with monitoring."""
    logger.info("Initializing Enhanced Flask Server")
    
    # Initial health check
    health_results = health_monitor.run_all_checks()
    critical_issues = [name for name, result in health_results.items() 
                      if result.status.value == "critical"]
    
    if critical_issues:
        logger.warning(f"Critical health issues detected: {critical_issues}")
        for issue in critical_issues:
            logger.error(f"Critical: {health_results[issue].message}")
    else:
        logger.success("All health checks passed")
    
    logger.info("Enhanced Flask Server initialized successfully")

def shutdown_server():
    """Graceful server shutdown."""
    logger.info("Shutting down Enhanced Flask Server")
    
    # Generate final performance report
    try:
        report_path = profiler.generate_html_report("final_session_report.html")
        logger.success(f"Final performance report generated: {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate final report: {str(e)}")
    
    logger.info("Enhanced Flask Server shutdown complete")

# Initialize on module load
initialize_server()

if __name__ == "__main__":
    try:
        logger.info("Starting Enhanced Flask Server in development mode")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        shutdown_server()
