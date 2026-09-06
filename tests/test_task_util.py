"""task_util 任务状态追踪单测"""

from knowledge.utils.task_util import (
    add_done_task,
    add_running_task,
    clear_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)


def test_running_then_done():
    add_running_task("t1", "node_a")
    assert "node_a" in get_running_task_list("t1")

    add_done_task("t1", "node_a")
    assert "node_a" not in get_running_task_list("t1")
    assert "node_a" in get_done_task_list("t1")


def test_no_duplicate_entries():
    add_running_task("t2", "node_a")
    add_running_task("t2", "node_a")
    assert get_running_task_list("t2").count("node_a") == 1


def test_status_and_result():
    update_task_status("t3", "processing")
    assert get_task_status("t3") == "processing"
    set_task_result("t3", "answer", "42")
    assert get_task_result("t3", "answer") == "42"
    assert get_task_result("t3", "missing", default="d") == "d"


def test_clear_task_removes_state():
    add_running_task("t4", "node_a")
    add_done_task("t4", "node_a")
    update_task_status("t4", "processing")
    clear_task("t4")
    assert get_task_status("t4") == ""
    assert get_running_task_list("t4") == []
    assert get_done_task_list("t4") == []


def test_unknown_node_name_passthrough():
    add_running_task("t5", "some_future_node")
    assert "some_future_node" in get_running_task_list("t5")
