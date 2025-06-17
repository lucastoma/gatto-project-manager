from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from .models import CollectorModel
from django.contrib.auth.decorators import login_required

# Enhanced views with authentication
@login_required
def collector_list(request):
    collectors = CollectorModel.objects.filter(owner=request.user)
    return render(request, 'collector/list.html', {'collectors': collectors})

@login_required
def collector_detail(request, pk):
    collector = get_object_or_404(CollectorModel, pk=pk, owner=request.user)
    return render(request, 'collector/detail.html', {'collector': collector})

def collector_api(request):
    collectors = list(CollectorModel.objects.values())
    return JsonResponse({'collectors': collectors, 'count': len(collectors)})
from django.http import JsonResponse
from .models import CollectorModel

# Create your views here.
def collector_list(request):
    collectors = CollectorModel.objects.all()
    return render(request, 'collector/list.html', {'collectors': collectors})

def collector_api(request):
    collectors = list(CollectorModel.objects.values())
    return JsonResponse({'collectors': collectors})
