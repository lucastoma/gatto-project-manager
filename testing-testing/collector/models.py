from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Enhanced models with better functionality
class CollectorModel(models.Model):
    name = models.CharField(max_length=100, help_text='Name of the collector')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.BooleanField(default=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='collectors')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Collector'
        verbose_name_plural = 'Collectors'
    
    def __str__(self):
        return f'{self.name} ({self.owner.username})'

class AnotherModel(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    priority = models.IntegerField(default=1)

# Create your models here.
class CollectorModel(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name

class AnotherModel(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
