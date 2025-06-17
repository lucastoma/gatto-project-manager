from django.urls import path
from . import views

app_name = 'collector'

urlpatterns = [
    # Strona główna
    path('', views.home, name='home'),
    
    # Projekty
    path('projects/', views.ProjectListView.as_view(), name='project_list'),
    path('projects/create/', views.ProjectCreateView.as_view(), name='project_create'),
    path('projects/<int:pk>/', views.ProjectDetailView.as_view(), name='project_detail'),
    path('projects/<int:pk>/edit/', views.ProjectUpdateView.as_view(), name='project_edit'),
    path('projects/<int:pk>/delete/', views.ProjectDeleteView.as_view(), name='project_delete'),
    path('projects/<int:pk>/groups/', views.FileGroupManageView.as_view(), name='filegroup_manage'),
    
    # Quick Option
    path('quick/', views.quick_option_view, name='quick_option'),
    path('quick/configs/', views.QuickConfigListView.as_view(), name='quick_config_list'),
    path('quick/configs/<int:pk>/edit/', views.QuickConfigEditView.as_view(), name='quick_config_edit'),
    path('quick/configs/<int:pk>/run/', views.quick_config_run, name='quick_config_run'),
    
    # Historia
    path('history/', views.CollectionHistoryListView.as_view(), name='history_list'),
    path('history/<int:pk>/', views.CollectionHistoryView.as_view(), name='collection_history'),
    
    # AJAX endpoints
    path('ajax/browse-directory/', views.ajax_directory_browse, name='ajax_directory_browse'),
    path('ajax/preview-files/', views.ajax_preview_files, name='ajax_preview_files'),
    
    # Faza 4 - Zaawansowane funkcje (placeholders)
    path('copy-to-clipboard/', views.copy_to_clipboard_view, name='copy_to_clipboard'),
    path('drag-drop/', views.drag_drop_handler, name='drag_drop_handler'),
]
