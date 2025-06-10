/**
 * WebView Main JavaScript
 * Główne funkcje dla interfejsu WebView
 */

// Globalne zmienne
window.WebView = {
    config: {
        maxFileSize: 10 * 1024 * 1024, // 10MB
        allowedTypes: ['image/jpeg', 'image/png', 'image/jpg'],
        apiBaseUrl: '/api',
        webviewBaseUrl: '/webview'
    },
    state: {
        currentTask: null,
        uploadedFiles: {},
        lastResults: null
    }
};

// Utility Functions
class WebViewUtils {
    /**
     * Wyświetl komunikat użytkownikowi
     */
    static showMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;
        
        // Znajdź kontener na komunikaty lub wstaw na początku main
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Usuń komunikat po 5 sekundach
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
    
    /**
     * Walidacja pliku przed uploadem
     */
    static validateFile(file) {
        const errors = [];
        
        // Sprawdź typ pliku
        if (!WebView.config.allowedTypes.includes(file.type)) {
            errors.push(`Nieprawidłowy typ pliku. Dozwolone: ${WebView.config.allowedTypes.join(', ')}`);
        }
        
        // Sprawdź rozmiar
        if (file.size > WebView.config.maxFileSize) {
            const maxSizeMB = WebView.config.maxFileSize / (1024 * 1024);
            errors.push(`Plik zbyt duży. Maksymalny rozmiar: ${maxSizeMB}MB`);
        }
        
        return errors;
    }
    
    /**
     * Konwertuj plik do base64 dla podglądu
     */
    static fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    /**
     * Formatuj rozmiar pliku
     */
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Debounce function
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// File Upload Handler
class FileUploadHandler {
    constructor(dropZone, fileInput, previewContainer) {
        this.dropZone = dropZone;
        this.fileInput = fileInput;
        this.previewContainer = previewContainer;
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Drag and drop events
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));
        
        // Click to upload
        this.dropZone.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }
    
    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }
    
    async processFiles(files) {
        for (const file of files) {
            const errors = WebViewUtils.validateFile(file);
            
            if (errors.length > 0) {
                WebViewUtils.showMessage(errors.join(', '), 'error');
                continue;
            }
            
            try {
                await this.displayPreview(file);
                WebViewUtils.showMessage(`Plik ${file.name} został załadowany`, 'success');
            } catch (error) {
                WebViewUtils.showMessage(`Błąd podczas ładowania pliku: ${error.message}`, 'error');
            }
        }
    }
    
    async displayPreview(file) {
        const base64 = await WebViewUtils.fileToBase64(file);
        
        const previewHtml = `
            <div class="image-container">
                <img src="${base64}" alt="${file.name}" class="image-preview">
                <p><strong>${file.name}</strong> (${WebViewUtils.formatFileSize(file.size)})</p>
            </div>
        `;
        
        this.previewContainer.innerHTML = previewHtml;
        
        // Zapisz plik w stanie globalnym
        const fieldName = this.fileInput.name;
        WebView.state.uploadedFiles[fieldName] = file;
    }
}

// Parameter Manager
class ParameterManager {
    constructor(formElement) {
        this.form = formElement;
        this.setupValidation();
    }
    
    setupValidation() {
        // Walidacja w czasie rzeczywistym
        const inputs = this.form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('input', WebViewUtils.debounce(() => {
                this.validateField(input);
            }, 300));
        });
    }
    
    validateField(field) {
        const value = field.value;
        const fieldName = field.name;
        let isValid = true;
        let errorMessage = '';
        
        // Walidacja specyficzna dla typu pola
        switch (field.type) {
            case 'number':
                const min = parseFloat(field.min);
                const max = parseFloat(field.max);
                const numValue = parseFloat(value);
                
                if (isNaN(numValue)) {
                    isValid = false;
                    errorMessage = 'Wartość musi być liczbą';
                } else if (min !== undefined && numValue < min) {
                    isValid = false;
                    errorMessage = `Wartość musi być >= ${min}`;
                } else if (max !== undefined && numValue > max) {
                    isValid = false;
                    errorMessage = `Wartość musi być <= ${max}`;
                }
                break;
                
            case 'text':
                if (field.required && !value.trim()) {
                    isValid = false;
                    errorMessage = 'To pole jest wymagane';
                }
                break;
        }
        
        // Wyświetl błąd walidacji
        this.displayFieldError(field, isValid ? null : errorMessage);
        
        return isValid;
    }
    
    displayFieldError(field, errorMessage) {
        // Usuń poprzedni błąd
        const existingError = field.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Dodaj nowy błąd jeśli istnieje
        if (errorMessage) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'field-error';
            errorDiv.style.color = 'var(--error-color)';
            errorDiv.style.fontSize = '0.875rem';
            errorDiv.style.marginTop = '0.25rem';
            errorDiv.textContent = errorMessage;
            
            field.parentNode.appendChild(errorDiv);
            field.style.borderColor = 'var(--error-color)';
        } else {
            field.style.borderColor = 'var(--border-color)';
        }
    }
    
    validateForm() {
        const inputs = this.form.querySelectorAll('input, select, textarea');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    getFormData() {
        const formData = new FormData(this.form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        return data;
    }
}

// API Client
class APIClient {
    static async request(endpoint, options = {}) {
        const url = `${WebView.config.apiBaseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }
    
    static async processAlgorithm(algorithmId, files, parameters) {
        const formData = new FormData();
        
        // Dodaj pliki
        for (const [key, file] of Object.entries(files)) {
            formData.append(key, file);
        }
        
        // Dodaj parametry
        for (const [key, value] of Object.entries(parameters)) {
            formData.append(key, value);
        }
        
        return await this.request(`/process`, {
            method: 'POST',
            headers: {}, // Usuń Content-Type dla FormData
            body: formData
        });
    }
    
    static async getTaskStatus(taskId) {
        return await this.request(`/task/${taskId}`);
    }
}

// Task Monitor
class TaskMonitor {
    constructor(taskId, onUpdate, onComplete, onError) {
        this.taskId = taskId;
        this.onUpdate = onUpdate;
        this.onComplete = onComplete;
        this.onError = onError;
        this.interval = null;
        this.start();
    }
    
    start() {
        this.interval = setInterval(async () => {
            try {
                const status = await APIClient.getTaskStatus(this.taskId);
                
                if (status.status === 'completed') {
                    this.stop();
                    this.onComplete(status.result);
                } else if (status.status === 'failed') {
                    this.stop();
                    this.onError(status.error);
                } else {
                    this.onUpdate(status);
                }
            } catch (error) {
                this.stop();
                this.onError(error.message);
            }
        }, 1000); // Sprawdzaj co sekundę
    }
    
    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }
}

// Progress Bar
class ProgressBar {
    constructor(element) {
        this.element = element;
        this.bar = element.querySelector('.progress-bar');
    }
    
    setProgress(percentage) {
        this.bar.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }
    
    show() {
        this.element.classList.remove('hidden');
    }
    
    hide() {
        this.element.classList.add('hidden');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('WebView JavaScript initialized');
    
    // Initialize file upload handlers
    const uploadZones = document.querySelectorAll('.upload-area');
    uploadZones.forEach(zone => {
        const fileInput = zone.querySelector('input[type="file"]') || 
                         zone.parentNode.querySelector('input[type="file"]');
        const previewContainer = zone.parentNode.querySelector('.preview-container');
        
        if (fileInput && previewContainer) {
            new FileUploadHandler(zone, fileInput, previewContainer);
        }
    });
    
    // Initialize parameter forms
    const parameterForms = document.querySelectorAll('.parameter-form');
    parameterForms.forEach(form => {
        new ParameterManager(form);
    });
});

// Export for global access
window.WebViewUtils = WebViewUtils;
window.FileUploadHandler = FileUploadHandler;
window.ParameterManager = ParameterManager;
window.APIClient = APIClient;
window.TaskMonitor = TaskMonitor;
window.ProgressBar = ProgressBar;