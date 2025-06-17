from django.db import models
from django.contrib.auth.models import User
import json


class Project(models.Model):
    """Model reprezentujący projekt z konfiguracją agregacji plików"""
    name = models.CharField(max_length=200, verbose_name="Nazwa projektu")
    description = models.TextField(blank=True, verbose_name="Opis projektu")
    root_path = models.CharField(max_length=500, verbose_name="Ścieżka główna")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Ustawienia wyjścia
    output_filename = models.CharField(max_length=200, default="output", verbose_name="Nazwa pliku wyjściowego")
    output_style = models.CharField(
        max_length=10, 
        choices=[('md', 'Markdown'), ('xml', 'XML')],
        default='md',
        verbose_name="Format wyjścia"
    )
    
    class Meta:
        verbose_name = "Projekt"
        verbose_name_plural = "Projekty"
        ordering = ['-updated_at']
    
    def __str__(self):
        return self.name


class FileGroup(models.Model):
    """Model reprezentujący grupę plików w projekcie"""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='file_groups')
    name = models.CharField(max_length=200, verbose_name="Nazwa grupy")
    description = models.TextField(blank=True, verbose_name="Opis grupy")
    order = models.PositiveIntegerField(default=0, verbose_name="Kolejność")
    
    class Meta:
        verbose_name = "Grupa plików"
        verbose_name_plural = "Grupy plików"
        ordering = ['order', 'name']
    
    def __str__(self):
        return f"{self.project.name} - {self.name}"


class GlobPattern(models.Model):
    """Model reprezentujący wzorzec glob dla grupy plików"""
    PATTERN_TYPE_CHOICES = [
        ('include', 'Dołącz'),
        ('exclude', 'Wyklucz'),
    ]
    
    file_group = models.ForeignKey(FileGroup, on_delete=models.CASCADE, related_name='patterns')
    pattern = models.CharField(max_length=500, verbose_name="Wzorzec glob")
    pattern_type = models.CharField(
        max_length=10, 
        choices=PATTERN_TYPE_CHOICES,
        default='include',
        verbose_name="Typ wzorca"
    )
    order = models.PositiveIntegerField(default=0, verbose_name="Kolejność")
    
    class Meta:
        verbose_name = "Wzorzec glob"
        verbose_name_plural = "Wzorce glob"
        ordering = ['order', 'pattern']
    
    def __str__(self):
        return f"{self.pattern} ({self.get_pattern_type_display()})"


class SearchPath(models.Model):
    """Model reprezentujący ścieżki przeszukiwania dla grupy plików"""
    file_group = models.ForeignKey(FileGroup, on_delete=models.CASCADE, related_name='search_paths')
    path = models.CharField(max_length=500, verbose_name="Ścieżka")
    recursive = models.BooleanField(default=True, verbose_name="Przeszukiwanie rekurencyjne")
    order = models.PositiveIntegerField(default=0, verbose_name="Kolejność")
    
    class Meta:
        verbose_name = "Ścieżka przeszukiwania"
        verbose_name_plural = "Ścieżki przeszukiwania"
        ordering = ['order', 'path']
    
    def __str__(self):
        return self.path


class QuickConfig(models.Model):
    """Model reprezentujący konfigurację quick option (.gatto-Q)"""
    directory_path = models.CharField(max_length=500, unique=True, verbose_name="Ścieżka katalogu")
    last_used_directory = models.CharField(max_length=500, blank=True, verbose_name="Ostatnio używany katalog")
    
    # Listy glob jako JSON
    code_patterns = models.JSONField(default=list, verbose_name="Wzorce kodu")
    doc_patterns = models.JSONField(default=list, verbose_name="Wzorce dokumentacji")
    concept_patterns = models.JSONField(default=list, verbose_name="Wzorce konceptów")
    user_patterns = models.JSONField(default=list, verbose_name="Wzorce użytkownika")
    
    # Globalne ustawienia include/exclude
    global_include_patterns = models.JSONField(default=list, verbose_name="Globalne wzorce dołączania")
    global_exclude_patterns = models.JSONField(default=list, verbose_name="Globalne wzorce wykluczania")
    
    # Ustawienia
    root_directory = models.CharField(max_length=500, blank=True, verbose_name="Katalog główny")
    output_format = models.CharField(
        max_length=10,
        choices=[('md', 'Markdown'), ('xml', 'XML')],
        default='md',
        verbose_name="Format wyjścia"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Konfiguracja Quick"
        verbose_name_plural = "Konfiguracje Quick"
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Quick Config: {self.directory_path}"
    
    def get_all_patterns(self):
        """Zwraca wszystkie wzorce jako słownik"""
        return {
            'code': self.code_patterns,
            'doc': self.doc_patterns,
            'concept': self.concept_patterns,
            'user': self.user_patterns,
            'global_include': self.global_include_patterns,
            'global_exclude': self.global_exclude_patterns,
        }


class CollectionHistory(models.Model):
    """Model przechowujący historię uruchomień agregacji"""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True, related_name='history')
    quick_config = models.ForeignKey(QuickConfig, on_delete=models.CASCADE, null=True, blank=True, related_name='history')
    
    output_file_path = models.CharField(max_length=500, verbose_name="Ścieżka pliku wyjściowego")
    files_processed = models.PositiveIntegerField(default=0, verbose_name="Liczba przetworzonych plików")
    execution_time = models.FloatField(null=True, blank=True, verbose_name="Czas wykonania (s)")
    success = models.BooleanField(default=True, verbose_name="Sukces")
    error_message = models.TextField(blank=True, verbose_name="Komunikat błędu")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Historia agregacji"
        verbose_name_plural = "Historia agregacji"
        ordering = ['-created_at']
    
    def __str__(self):
        config_name = self.project.name if self.project else f"Quick: {self.quick_config.directory_path}"
        return f"{config_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
