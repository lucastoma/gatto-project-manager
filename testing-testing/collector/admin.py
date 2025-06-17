from django.contrib import admin
from .models import CollectorModel, AnotherModel, ThirdModel

# Register all models here
admin.site.register(CollectorModel)
admin.site.register(AnotherModel)
admin.site.register(ThirdModel)

# Enhanced admin configuration
class CollectorModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'status', 'priority']
    search_fields = ['name', 'description']
    list_filter = ['status', 'created_at']
from .models import CollectorModel, AnotherModel

# Register your models here.
admin.site.register(CollectorModel)
admin.site.register(AnotherModel)

# Additional admin configuration
class CollectorModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'status']
    search_fields = ['name']
