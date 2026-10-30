"""The same constructive families with no initial examples, in both source arms."""
import sys
from witness_task_search import register


def main():
    names=register();mode=sys.argv.pop(1)
    if mode=='prepare':
        from prepare_witness_zero_task_search import main as prepare
        prepare(names)
    else:
        assert mode=='run'
        from run_task_search import main as run
        run()


if __name__=='__main__': main()
