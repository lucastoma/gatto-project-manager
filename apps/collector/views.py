from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import os
import json
from pathlib import Path

from .models import Project, FileGroup, QuickConfig, CollectionHistory
from .forms import (
    ProjectForm, FileGroupForm, QuickConfigForm, DirectorySelectionForm,
    GlobPatternFormSet, SearchPathFormSet, FileGroupFormSet
)
from .quick_context_collector import QuickContextCollectorApp
# from .advanced_context_collector import AdvancedContextCollector  # Not needed for now


def home(request):
    """Strona główna aplikacji"""
    recent_projects = Project.objects.all()[:5]
    recent_quick_configs = QuickConfig.objects.all()[:5]
    recent_history = CollectionHistory.objects.all()[:10]
    
    context = {
        'recent_projects': recent_projects,
        'recent_quick_configs': recent_quick_configs,
        'recent_history': recent_history,
    }
    return render(request, 'collector/home.html', context)


class ProjectListView(ListView):
    """Lista projektów"""
    model = Project
    template_name = 'collector/project_list.html'
    context_object_name = 'projects'
    paginate_by = 10


class ProjectDetailView(DetailView):
    """Szczegóły projektu"""
    model = Project
    template_name = 'collector/project_detail.html'
    context_object_name = 'project'


class ProjectCreateView(CreateView):
    """Tworzenie nowego projektu"""
    model = Project
    form_class = ProjectForm
    template_name = 'collector/project_form.html'
    success_url = reverse_lazy('collector:project_list')
    
    def form_valid(self, form):
        if self.request.user.is_authenticated:
            form.instance.owner = self.request.user
        messages.success(self.request, 'Projekt został utworzony pomyślnie!')
        return super().form_valid(form)


class ProjectUpdateView(UpdateView):
    """Edycja projektu"""
    model = Project
    form_class = ProjectForm
    template_name = 'collector/project_form.html'
    
    def get_success_url(self):
        return reverse_lazy('collector:project_detail', kwargs={'pk': self.object.pk})
    
    def form_valid(self, form):
        messages.success(self.request, 'Projekt został zaktualizowany!')
        return super().form_valid(form)


class ProjectDeleteView(DeleteView):
    """Usuwanie projektu"""
    model = Project
    template_name = 'collector/project_confirm_delete.html'
    success_url = reverse_lazy('collector:project_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Projekt został usunięty!')
        return super().delete(request, *args, **kwargs)


@method_decorator(csrf_exempt, name='dispatch')
class FileGroupManageView(UpdateView):
    """Zarządzanie grupami plików w projekcie"""
    model = Project
    template_name = 'collector/filegroup_manage.html'
    form_class = ProjectForm
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['filegroup_formset'] = FileGroupFormSet(self.request.POST, instance=self.object)
        else:
            context['filegroup_formset'] = FileGroupFormSet(instance=self.object)
        return context
    
    def form_valid(self, form):
        context = self.get_context_data()
        filegroup_formset = context['filegroup_formset']
        
        if filegroup_formset.is_valid():
            self.object = form.save()
            filegroup_formset.instance = self.object
            filegroup_formset.save()
            messages.success(self.request, 'Grupy plików zostały zaktualizowane!')
            return redirect('collector:project_detail', pk=self.object.pk)
        else:
            return self.render_to_response(self.get_context_data(form=form))


def quick_option_view(request):
    """Widok dla quick option - szybka agregacja jednego katalogu"""
    form = DirectorySelectionForm()
    quick_config = None
    
    if request.method == 'POST':
        form = DirectorySelectionForm(request.POST)
        if form.is_valid():
            directory = form.cleaned_data['directory']
            use_last = form.cleaned_data.get('use_last_directory', False)
            
            # Sprawdź czy istnieje konfiguracja dla tego katalogu
            quick_config, created = QuickConfig.objects.get_or_create(
                directory_path=directory,
                defaults={
                    'last_used_directory': directory,
                    'root_directory': directory,
                    'code_patterns': ['*.py', '*.js', '*.tsx', '*.jsx'],
                    'doc_patterns': ['*.md', '*.txt', '*.rst'],
                    'concept_patterns': [],
                    'user_patterns': [],
                    'global_include_patterns': ['*.*'],
                    'global_exclude_patterns': ['node_modules/**', '.git/**', '*.pyc', '__pycache__/**'],
                }
            )
            
            if use_last and quick_config.last_used_directory:
                directory = quick_config.last_used_directory
            
            return redirect('collector:quick_config_edit', pk=quick_config.pk)
    
    # Pokaż ostatnie konfiguracje
    recent_configs = QuickConfig.objects.all()[:5]
    
    context = {
        'form': form,
        'recent_configs': recent_configs,
    }
    return render(request, 'collector/quick_option.html', context)


class QuickConfigEditView(UpdateView):
    """Edycja konfiguracji quick option"""
    model = QuickConfig
    form_class = QuickConfigForm
    template_name = 'collector/quick_config_form.html'
    
    def get_success_url(self):
        return reverse_lazy('collector:quick_config_run', kwargs={'pk': self.object.pk})
    
    def form_valid(self, form):
        messages.success(self.request, 'Konfiguracja została zaktualizowana!')
        return super().form_valid(form)


def quick_config_run(request, pk):
    """Uruchomienie agregacji dla konfiguracji quick"""
    quick_config = get_object_or_404(QuickConfig, pk=pk)
    
    if request.method == 'POST':
        try:
            # Tu będzie logika uruchamiania quick_context_collector
            # Na razie symulacja
            output_path = os.path.join(quick_config.directory_path, f'quick_output.{quick_config.output_format}')
            
            # Zapisz w historii
            history = CollectionHistory.objects.create(
                quick_config=quick_config,
                output_file_path=output_path,
                files_processed=42,  # Placeholder
                execution_time=1.5,  # Placeholder
                success=True
            )
            
            messages.success(request, f'Agregacja zakończona! Plik: {output_path}')
            return redirect('collector:collection_history', pk=history.pk)
            
        except Exception as e:
            messages.error(request, f'Błąd podczas agregacji: {str(e)}')
    
    context = {
        'quick_config': quick_config,
    }
    return render(request, 'collector/quick_config_run.html', context)


class QuickConfigListView(ListView):
    """Lista konfiguracji quick"""
    model = QuickConfig
    template_name = 'collector/quick_config_list.html'
    context_object_name = 'configs'
    paginate_by = 10


class CollectionHistoryView(DetailView):
    """Szczegóły historii agregacji"""
    model = CollectionHistory
    template_name = 'collector/collection_history.html'
    context_object_name = 'history'


class CollectionHistoryListView(ListView):
    """Lista historii agregacji"""
    model = CollectionHistory
    template_name = 'collector/collection_history_list.html'
    context_object_name = 'history_list'
    paginate_by = 20
    ordering = ['-created_at']


@csrf_exempt
def ajax_directory_browse(request):
    """AJAX endpoint do przeglądania katalogów"""
    if request.method == 'POST':
        data = json.loads(request.body)
        path = data.get('path', '.')
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return JsonResponse({'error': 'Ścieżka nie istnieje'}, status=400)
            
            if not path_obj.is_dir():
                return JsonResponse({'error': 'To nie jest katalog'}, status=400)
            
            directories = []
            for item in path_obj.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    directories.append({
                        'name': item.name,
                        'path': str(item),
                    })
            
            directories.sort(key=lambda x: x['name'])
            
            return JsonResponse({
                'directories': directories,
                'current_path': str(path_obj.absolute()),
                'parent_path': str(path_obj.parent) if path_obj.parent != path_obj else None,
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Metoda nie dozwolona'}, status=405)


@csrf_exempt
def ajax_preview_files(request):
    """AJAX endpoint do podglądu plików pasujących do wzorców"""
    if request.method == 'POST':
        data = json.loads(request.body)
        directory = data.get('directory', '.')
        patterns = data.get('patterns', [])
        exclude_patterns = data.get('exclude_patterns', [])
        
        try:
            # Tu będzie logika podglądu plików
            # Na razie placeholder
            files = [
                {'path': 'example.py', 'size': 1024},
                {'path': 'test.js', 'size': 512},
            ]
            
            return JsonResponse({
                'files': files,
                'total_count': len(files),
                'total_size': sum(f['size'] for f in files),
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Metoda nie dozwolona'}, status=405)


def copy_to_clipboard_view(request):
    """Endpoint do kopiowania zawartości do schowka (Faza 4)"""
    # Placeholder dla Fazy 4
    return JsonResponse({'message': 'Funkcja będzie dostępna w Fazie 4'})


def drag_drop_handler(request):
    """Handler dla drag & drop plików (Faza 4)"""
    # Placeholder dla Fazy 4
    return JsonResponse({'message': 'Funkcja będzie dostępna w Fazie 4'})
