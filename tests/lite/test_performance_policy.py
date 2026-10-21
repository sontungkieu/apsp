import pytest
from apsp.lite.performance import ExecutionPolicy
from apsp.lite.collector import runtime_identity,task_contract


def test_execution_changes_are_bound_into_resume_contract():
    base={'source_hash':'abc','hostname':'node1','source_commit':'git1'}
    fp1={**base,'execution':ExecutionPolicy(decode_batch=1).to_dict()}
    fp2={**base,'execution':ExecutionPolicy(decode_batch=4).to_dict()}
    assert task_contract({'task_id':'same'},fp1)!=task_contract({'task_id':'same'},fp2)
    assert runtime_identity(fp1)==runtime_identity({**fp1,'hostname':'node2','source_commit':'git2'})


@pytest.mark.parametrize('kwargs',[{'decode_batch':0},{'decode_batch':3},{'score_batch':True},{'capture_on_gpu':'yes'},{'schema':'unknown'},
    {'decode_batch':2,'decode_streams':2},{'decode_streams':3},{'png_compress_level':10},{'fuse_checkpoints':'yes'}])
def test_invalid_batch_policy_fails_before_gpu_allocation(kwargs):
    with pytest.raises(ValueError):ExecutionPolicy(**kwargs)
