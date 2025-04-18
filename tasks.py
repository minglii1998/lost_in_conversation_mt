# from task_base import Task
from task_database import TaskDatabase
from task_math import TaskMath
from task_python import TaskPython
from task_summary import TaskSummary
from task_data2text import TaskData2Text
from task_apis import TaskAPIs
from task_translation import TaskTranslation


def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("database"):
        return TaskDatabase(**kwargs)
    elif task_name == "python":
        return TaskPython(**kwargs)
    elif task_name == "translation":
        return TaskTranslation(**kwargs)
    elif task_name == "summary":
        return TaskSummary(**kwargs)
    elif task_name == "data2text":
        return TaskData2Text(**kwargs)
    elif task_name == "math":
        return TaskMath(**kwargs)
    elif task_name.startswith("apis"):
        return TaskAPIs(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")

