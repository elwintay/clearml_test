from clearml import Task, StorageManager, Dataset
import json
import os
import shutil
import argparse
from pathlib import Path

print("I AM HERE")
task = Task.init(project_name='GTT-test', task_name='baseGTT-1', output_uri="s3://experiment-logging/storage/")
print("I AM HERE 2")
task.set_base_docker("nvcr.io/nvidia/pytorch:20.06-py3")
task.execute_remotely(queue_name="elwin", exit_process=True)