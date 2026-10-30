"""Independent audits and paired inference for feedback tracking experiments."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from analyze_training import estimate
from tracking_task import HORIZON, STRATA, Streams

ARMS = ("expert", "on_policy", "random", "frozen_policy")
LABELS = {"expert": "Expert demonstrations", "on_policy": "Current-model exploration", "random": "Random exploration", "frozen_policy": "Frozen-model exploration"}


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(line) for line in path.read_text().splitlines()]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_grade(episode, actions):
    assert len(actions) == len(episode["commands"]) == 16
    error, errors, optimal = episode["initial_error"], [], []
    for command, action in zip(episode["commands"], actions, strict=True):
        choices = [abs(error + value - command) for value in range(-3, 4)]
        assert 0 <= action < 7
        optimal.append(action == choices.index(min(choices)))
        error += action - 3 - command
        errors.append(error)
    return {"errors": errors, "optimal": optimal, "mean_absolute_error": sum(abs(value) for value in errors) / 16,
            "stable_tracking": all(value == 0 for value in errors[-8:]), "final_error": abs(errors[-1])}


def audit_rows(rows, stream, policy):
    assert len(rows) % stream.lanes == 0
    for offset in range(0, len(rows), stream.lanes):
        chunk = rows[offset:offset + stream.lanes]
        expected = stream.current()
        for row, reference in zip(chunk, expected, strict=True):
            assert {key: row[key] for key in ('u', 'e', 'age')} == {key: reference[key] for key in ('u', 'e', 'age')}
            possible = [abs(row['e'] + action - row['u']) for action in range(-3, 4)]
            assert row['target'] == possible.index(min(possible))
            if policy == 'expert': assert row['e'] == 0 and row['action'] == row['target']
        stream.advance([row['action'] for row in chunk])


def load_run(directory):
    config, dataset = read(directory/'config.json'), read(directory/'dataset.json')
    assert digest(directory/'dataset.json') == config['dataset_sha256']
    assert {name:digest(directory/'source'/name) for name in config['source_sha256']} == config['source_sha256']
    warm = read(directory/'warm_training.json')
    audit_rows(warm, Streams(config['seed']), 'expert')
    records, diagnostics, audits, command_streams = {}, {}, {}, []
    for arm in ARMS:
        logs = lines(directory/f'{arm}_training.jsonl')
        final = read(directory/f'{arm}_final.json')
        stream = Streams(config['seed'] + 30000)
        history, seen, gram, commands, rounds = list(warm), set(), np.zeros((2,2)), [], []
        for index, log in enumerate(logs):
            budget = (index+1)*512
            assert log['budget'] == budget and len(log['rows']) == 512
            audit_rows(log['rows'], stream, arm)
            history.extend(log['rows'])
            assert len(log['replay_indices']) == 512
            assert all(0 <= value < len(history) for value in log['replay_indices'])
            assert sorted(log['order']) == list(range(1024))
            count = log['counters']
            assert count['labels_acquired'] == budget
            assert count['training_examples'] == count['supervised_tokens'] == 2*budget
            assert count['optimizer_updates'] == 2*budget//64
            inputs = np.array([[row['u'],row['e']] for row in log['rows']])
            gram += inputs.T @ inputs
            commands.extend(row['u'] for row in log['rows'])
            seen.update((row['u'],row['e']) for row in log['rows'])
            rounds.append({'budget':budget,'unique_inputs':len(seen),'nonzero_error_fraction':float(np.mean(inputs[:,1] != 0)),
                           'minimum_input_gram_eigenvalue':float(np.linalg.eigvalsh(gram)[0]),
                           'mean_squared_input_error':float(np.mean(inputs[:,1]**2)), 'loss':log['loss'],
                           'action_counts':dict(Counter(row['action'] for row in log['rows']))})
        command_streams.append(commands)
        assert logs[-1]['budget'] == config['labels'] and final['counters'] == logs[-1]['counters']
        if arm == 'frozen_policy': assert read(directory/'frozen_acquisition.json')['rows'] == [row for log in logs for row in log['rows']]
        metrics = lines(directory/f'{arm}_metrics.jsonl')
        assert [row['budget'] for row in metrics] == config['checkpoints']
        for record in metrics:
            predictions = read(directory/f"{arm}_n{record['budget']}_predictions.json")
            expected = [(stratum,episode) for stratum,episodes in dataset['eval'].items() for episode in episodes]
            assert len(predictions) == len(expected)
            for row,(stratum,episode) in zip(predictions,expected,strict=True):
                assert row['stratum'] == stratum and row['episode'] == episode
                grade = independently_grade(episode,row['actions'])
                assert all(row[key] == value for key,value in grade.items())
            graded = {}
            for stratum in STRATA:
                rows = [row for row in predictions if row['stratum'] == stratum]
                graded[stratum] = {field:float(np.mean([row[field] for row in rows])) for field in ('mean_absolute_error','stable_tracking','final_error')}
                graded[stratum]['action_accuracy'] = float(np.mean([value for row in rows for value in row['optimal']]))
                graded[stratum]['count'] = len(rows)
            assert record['metrics'] == graded and record['seed'] == config['seed'] and record['arm'] == arm
            if record['budget']: assert record['counters'] == logs[record['budget']//512-1]['counters']
        records[arm], diagnostics[arm] = metrics, rounds
        audits[arm] = {'labels_verified':config['labels'],'trajectories_verified':len(metrics)*len(expected),'all_checkpoints_present':True}
    assert all(commands == command_streams[0] for commands in command_streams)
    training_sequences=set()
    for values in ([row['u'] for row in warm],command_streams[0]):
        assert len(values)%(128*16)==0
        for start in range(0,len(values)//128,16):
            for lane in range(128): training_sequences.add(tuple(values[(start+step)*128+lane] for step in range(16)))
    test_sequences={tuple(episode['commands']) for episodes in dataset['eval'].values() for episode in episodes}
    assert not training_sequences.intersection(test_sequences), 'A fresh evaluation command sequence occurred in training.'
    baseline = read(directory/'expert_n0_predictions.json')
    assert all(read(directory/f'{arm}_n0_predictions.json') == baseline for arm in ARMS)
    audits['sequence_isolation']={'training_sequences':len(training_sequences),'test_sequences':len(test_sequences),'overlap':0}
    return {'seed':config['seed'],'config':config,'dataset':dataset,'records':records,'diagnostics':diagnostics,'audits':audits}


def analyze(runs, output):
    loaded = [load_run(run) for run in tqdm(runs,desc='Audit tracking states and trajectories')]
    assert len({run['seed'] for run in loaded}) == len(loaded)
    first = loaded[0]
    oracle_metrics = {}
    for stratum,episodes in first['dataset']['eval'].items():
        oracle_rows = []
        for episode in episodes:
            error,actions = episode['initial_error'],[]
            for command in episode['commands']:
                choices = [abs(error+a-command) for a in range(-3,4)]
                action = choices.index(min(choices))
                actions.append(action)
                error += action-3-command
            oracle_rows.append(independently_grade(episode,actions))
        oracle_metrics[stratum] = {'mean_absolute_error':float(np.mean([r['mean_absolute_error'] for r in oracle_rows])),
                                   'stable_tracking':float(np.mean([r['stable_tracking'] for r in oracle_rows])),
                                   'feedforward_mean_absolute_error':float(np.mean([abs(e['initial_error']) for e in episodes]))}
    for run in loaded[1:]:
        assert run['dataset']['eval'] == first['dataset']['eval']
        for key in ('labels','learning_rate','warm_labels','warm_epochs','source_sha256','checkpoints'): assert run['config'][key] == first['config'][key]
    curves = {}
    for arm in ARMS:
        curves[arm] = []
        for index,budget in enumerate(first['config']['checkpoints']):
            rows = [run['records'][arm][index] for run in loaded]
            metrics = {stratum:{field:estimate([row['metrics'][stratum][field] for row in rows]) for field in ('mean_absolute_error','stable_tracking','action_accuracy','final_error')} for stratum in STRATA}
            clocks = {'labels':estimate([budget]*len(rows)),
                      'phase_seconds':estimate([row['counters']['training_seconds']+row['counters']['generation_seconds'] for row in rows]),
                      'tokens':estimate([sum(row['counters'][key] for key in ('training_prompt_tokens','supervised_tokens','generation_prompt_tokens','generated_actions')) for row in rows])}
            curves[arm].append({'budget':budget,'metrics':metrics,'clocks':clocks})
    contrasts = [{'first':'on_policy','second':other,'stratum':stratum,'metric':metric,
                  'difference':estimate([run['records']['on_policy'][-1]['metrics'][stratum][metric]-run['records'][other][-1]['metrics'][stratum][metric] for run in loaded])}
                 for other in ('expert','random','frozen_policy') for stratum in STRATA for metric in ('mean_absolute_error','stable_tracking')]
    summary = {'complete':True,'seeds':[run['seed'] for run in loaded],'budget':first['config']['labels'],'model':first['config']['model'],
               'curves':curves,'endpoint_contrasts':contrasts,'oracle_metrics':oracle_metrics,'audits':{run['seed']:run['audits'] for run in loaded},
               'diagnostics':{run['seed']:run['diagnostics'] for run in loaded},
               'uncertainty':'Paired training-seed bootstrap, conditional on common fresh command sequences; single-seed pilot has no interval. No power-law exponent fitted.',
               'interpretation':'Nominal training trajectories start e0=0. Recovery and large-shift evaluations deliberately perturb initial error. Individual Markov inputs can repeat between training and fresh episodes. Exploration includes an explicit10%uniform action component.'}
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for stratum in STRATA:
        for clock in ('labels','phase_seconds'):
            figure,axis = plt.subplots()
            for arm in ARMS:
                rows = curves[arm]
                x = [row['clocks'][clock]['mean'] for row in rows]
                y = [row['metrics'][stratum]['mean_absolute_error']['mean'] for row in rows]
                axis.plot(x,y,marker='o',label=LABELS[arm])
                if len(loaded)>1:
                    bounds = np.array([row['metrics'][stratum]['mean_absolute_error']['ci95'] for row in rows])
                    axis.fill_between(x,bounds[:,0],bounds[:,1],alpha=0.15)
            axis.set(xlabel='Oracle state labels acquired' if clock=='labels' else 'Training + acquisition seconds',ylabel='Mean absolute tracking error',title=stratum.replace('_',' '),ylim=(0,None))
            axis.axhline(oracle_metrics[stratum]['mean_absolute_error'],color='black',linestyle=':',label='Exact oracle')
            axis.legend()
            figure.tight_layout()
            path = output/f'{stratum}_{clock}.png'
            figure.savefig(path)
            plt.close(figure)
            paths.append(str(path.resolve()))
    summary['plots'] = paths
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for path in paths: print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    load_dotenv(Path.home()/'.env')
    os.environ['MPLCONFIGDIR'] = str(Path(os.environ['FAST_CACHE_DIR'])/'generative-learning/matplotlib')
    analyze(args.runs,args.output)


if __name__ == '__main__': main()
