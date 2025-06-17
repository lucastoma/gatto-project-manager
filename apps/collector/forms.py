from django import forms
from django.forms import inlineformset_factory
from .models import Project, FileGroup, GlobPattern, SearchPath, QuickConfig


class ProjectForm(forms.ModelForm):
    """Formularz do tworzenia i edycji projektów"""
    
    class Meta:
        model = Project
        fields = ['name', 'description', 'root_path', 'output_filename', 'output_style']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nazwa projektu'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Opis projektu'}),
            'root_path': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '/ścieżka/do/projektu'}),
            'output_filename': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'output'}),
            'output_style': forms.Select(attrs={'class': 'form-control'}),
        }


class FileGroupForm(forms.ModelForm):
    """Formularz do tworzenia i edycji grup plików"""
    
    class Meta:
        model = FileGroup
        fields = ['name', 'description', 'order']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nazwa grupy'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'Opis grupy'}),
            'order': forms.NumberInput(attrs={'class': 'form-control', 'min': 0}),
        }


class GlobPatternForm(forms.ModelForm):
    """Formularz do zarządzania wzorcami glob"""
    
    class Meta:
        model = GlobPattern
        fields = ['pattern', 'pattern_type', 'order']
        widgets = {
            'pattern': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': '*.py, **/*.js, etc.',
                'data-toggle': 'tooltip',
                'title': 'Wzorzec glob (np. *.py, **/*.js, src/**/*.tsx)'
            }),
            'pattern_type': forms.Select(attrs={'class': 'form-control'}),
            'order': forms.NumberInput(attrs={'class': 'form-control', 'min': 0}),
        }


class SearchPathForm(forms.ModelForm):
    """Formularz do zarządzania ścieżkami przeszukiwania"""
    
    class Meta:
        model = SearchPath
        fields = ['path', 'recursive', 'order']
        widgets = {
            'path': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': '. lub src/ lub /absolute/path',
                'data-toggle': 'tooltip',
                'title': 'Ścieżka względna lub bezwzględna do przeszukania'
            }),
            'recursive': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'order': forms.NumberInput(attrs={'class': 'form-control', 'min': 0}),
        }


class QuickConfigForm(forms.ModelForm):
    """Formularz do konfiguracji quick option"""
    
    # Dodatkowe pola dla łatwiejszej edycji wzorców
    code_patterns_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '*.py\n*.js\n*.tsx'
        }),
        label="Wzorce kodu (jeden na linię)",
        help_text="Wzorce plików kodu źródłowego"
    )
    
    doc_patterns_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '*.md\n*.txt\n*.rst'
        }),
        label="Wzorce dokumentacji (jeden na linię)",
        help_text="Wzorce plików dokumentacji"
    )
    
    concept_patterns_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '*.concept\n*.idea\n*.design'
        }),
        label="Wzorce konceptów (jeden na linię)",
        help_text="Wzorce plików konceptualnych"
    )
    
    user_patterns_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '*.custom\n*.user'
        }),
        label="Wzorce użytkownika (jeden na linię)",
        help_text="Niestandardowe wzorce użytkownika"
    )
    
    global_include_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2,
            'placeholder': '*.*\n**/*'
        }),
        label="Globalne wzorce dołączania (jeden na linię)",
        help_text="Wzorce plików do zawsze dołączenia"
    )
    
    global_exclude_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'node_modules/**\n.git/**\n*.pyc'
        }),
        label="Globalne wzorce wykluczania (jeden na linię)",
        help_text="Wzorce plików do zawsze wykluczenia"
    )
    
    class Meta:
        model = QuickConfig
        fields = [
            'directory_path', 'last_used_directory', 'root_directory', 'output_format'
        ]
        widgets = {
            'directory_path': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': '/ścieżka/do/katalogu',
                'readonly': True
            }),
            'last_used_directory': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': '/ostatnio/używany/katalog'
            }),
            'root_directory': forms.TextInput(attrs={
                'class': 'form-control', 
                'placeholder': '/katalog/główny/projektu'
            }),
            'output_format': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Wypełnij pola tekstowe z JSON
        if self.instance and self.instance.pk:
            self.fields['code_patterns_text'].initial = '\n'.join(self.instance.code_patterns)
            self.fields['doc_patterns_text'].initial = '\n'.join(self.instance.doc_patterns)
            self.fields['concept_patterns_text'].initial = '\n'.join(self.instance.concept_patterns)
            self.fields['user_patterns_text'].initial = '\n'.join(self.instance.user_patterns)
            self.fields['global_include_text'].initial = '\n'.join(self.instance.global_include_patterns)
            self.fields['global_exclude_text'].initial = '\n'.join(self.instance.global_exclude_patterns)
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        # Konwertuj pola tekstowe na listy JSON
        def text_to_list(text):
            if not text:
                return []
            return [line.strip() for line in text.split('\n') if line.strip()]
        
        instance.code_patterns = text_to_list(self.cleaned_data.get('code_patterns_text', ''))
        instance.doc_patterns = text_to_list(self.cleaned_data.get('doc_patterns_text', ''))
        instance.concept_patterns = text_to_list(self.cleaned_data.get('concept_patterns_text', ''))
        instance.user_patterns = text_to_list(self.cleaned_data.get('user_patterns_text', ''))
        instance.global_include_patterns = text_to_list(self.cleaned_data.get('global_include_text', ''))
        instance.global_exclude_patterns = text_to_list(self.cleaned_data.get('global_exclude_text', ''))
        
        if commit:
            instance.save()
        return instance


class DirectorySelectionForm(forms.Form):
    """Formularz do wyboru katalogu dla quick option"""
    directory = forms.CharField(
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '/ścieżka/do/katalogu',
            'id': 'directory-input'
        }),
        label="Katalog do przetworzenia"
    )
    
    use_last_directory = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        label="Użyj ostatnio używanego katalogu"
    )


# Formsets dla zarządzania powiązanymi obiektami
GlobPatternFormSet = inlineformset_factory(
    FileGroup, 
    GlobPattern, 
    form=GlobPatternForm,
    extra=1,
    can_delete=True
)

SearchPathFormSet = inlineformset_factory(
    FileGroup, 
    SearchPath, 
    form=SearchPathForm,
    extra=1,
    can_delete=True
)

FileGroupFormSet = inlineformset_factory(
    Project, 
    FileGroup, 
    form=FileGroupForm,
    extra=1,
    can_delete=True
)