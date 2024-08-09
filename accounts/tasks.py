from celery import shared_task
import time

@shared_task
def long_running_task():
    time.sleep(300)  # Simulate long-running task
    return 'Task completed'