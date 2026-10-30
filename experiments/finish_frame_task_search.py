"""Run the frozen baseline-selected search over shared allocation-task wordings."""

from pathlib import Path
import subprocess
import sys


def main():
    project=Path(__file__).resolve().parents[1]; root=(project/'results/task_search').resolve(); code=project/'experiments'; selection=root/'frame_screen_selection.json'
    commands=[['framed_task_search.py','run','--phase','frame_calibration','--ns','0','--arms','--concurrency','2','--world-concurrency','2'],
              ['framed_task_search.py','select','--phase','frame_calibration','--output',str(selection)],
              ['framed_task_search.py','prepare','--phase','frame_screen','--selection',str(selection),'--worlds','4','--practice','8','--tests','8'],
              ['framed_task_search.py','run','--phase','frame_screen','--ns','0','2','8','--arms','teacher','self_fixed','--concurrency','4','--world-concurrency','2'],
              ['analyze_task_search.py','--phase','frame_screen','--endpoint','8','--confirmation-selection',str(root/'frame_confirmation_selection.json'),'--limit','2'],
              ['framed_task_search.py','audit','--phase','frame_screen'],['framed_task_search.py','strategy_audit','--phase','frame_screen'],
              ['verify_task_search.py','--phases','frame_screen']]
    for index,command in enumerate(commands):
        print(command,flush=True)
        with (root/f'frame_step_{index}.log').open('a') as log: subprocess.run([sys.executable,str(code/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Shared-wording screen completed; fresh confirmation is required for a selected candidate.',flush=True)


if __name__=='__main__': main()
