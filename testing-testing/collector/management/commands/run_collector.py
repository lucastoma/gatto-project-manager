from django.core.management.base import BaseCommand
from collector.models import CollectorModel

class Command(BaseCommand):
    help = 'Run collector operations'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting collector...')
        collectors = CollectorModel.objects.all()
        self.stdout.write(f'Found {collectors.count()} collectors')
        self.stdout.write('Collector finished successfully')