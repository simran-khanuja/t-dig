"""Multi-node service fleet: the policy on node zero, reward services on every node.

Every node has eight GPUs and one VQAScore process. Node zero gives two GPUs to the policy and
five to reward services; each further node gives seven to reward services. Nothing starts
without --execute.

Three environment variables say how to reach the other nodes:
  T_DIG_NODEFILE       file listing the hosts, one per line, node zero first (under Slurm,
                       default: scontrol show hostnames "$SLURM_JOB_NODELIST")
  T_DIG_REMOTE_LAUNCH  command prefix that runs a process on node {index} (0-based) or {host},
                       e.g. 'ssh {host}' (under Slurm, default:
                       'srun --nodes=1 --ntasks=1 -w {host} --')
  T_DIG_JOB_ID         job id, shared by the nodes of one run (default: $SLURM_JOB_ID, else
                       local-<pid>)
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request

from hpdv2.storage import atomic_json, read_json, scheduler_job_id


def allocation_hosts(nodefile, nodes):
    if nodefile:
        text = Path(nodefile).read_text()
    elif os.environ.get('SLURM_JOB_NODELIST'):
        text = subprocess.run(
            ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    else:
        raise ValueError('Set T_DIG_NODEFILE to a file listing the hosts, node zero first')
    hosts = list(dict.fromkeys(text.split()))
    if len(hosts) != nodes:
        raise ValueError(
            f'This experiment requires exactly {nodes} allocated nodes, got {len(hosts)}'
        )
    return hosts


def gpu_roles(node_index, visible):
    # Node 0: 2 policy GPUs, 1 VQAScore GPU, 5 reward GPUs. Further nodes: 1 VQAScore, 7 reward.
    if node_index < 0 or len(visible) != 8 or len(set(visible)) != 8:
        raise ValueError('Require a node role with eight distinct GPUs')
    trainer = visible[:2] if node_index == 0 else []
    return dict(trainer=trainer, vqa=visible[len(trainer)], renderers=visible[len(trainer) + 1 :])


def remote_command(cfg, config_path, session, index, host):
    launch = os.environ.get('T_DIG_REMOTE_LAUNCH')
    if not launch:
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError("Set T_DIG_REMOTE_LAUNCH, e.g. 'ssh {host}'")
        launch = 'srun --nodes=1 --ntasks=1 -w {host} --'
    return shlex.split(launch.format(index=index, host=host)) + [
        '/bin/bash',
        str(Path(cfg['source']) / 'scripts/training_fleet_node.sh'),
        cfg['source'],
        cfg['train_python'],
        str(Path(config_path).resolve()),
        str(index),
        session,
    ]


class Processes:
    def __init__(self, out, cfg):
        self.out, self.cfg = Path(out), cfg
        self.out.mkdir(parents=True, exist_ok=True)
        self.children, self.logs, self.services = [], [], []

    def start(self, name, cmd, gpus=None, service=False):
        env = dict(
            os.environ,
            PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',
            HF_HUB_CACHE=self.cfg['cache'],
            WANDB_MODE='offline',
        )
        if gpus is not None:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        env['PATH'] = str(Path(cmd[0]).parent) + os.pathsep + env.get('PATH', '')
        env['RAY_ADDRESS'] = 'local'
        env['VERL_USE_EXTERNAL_MODULES'] = 'verl_omni'
        env['RAY_TMPDIR'] = '/tmp/tdig-ray-' + scheduler_job_id().split('.')[0]
        log = (self.out / (name + '.log')).open('a')
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.logs.append(log)
        self.children.append(proc)
        if service:
            self.services.append(proc)
        return proc

    def check(self):
        for proc in self.services:
            if proc.poll() is not None:
                raise RuntimeError(
                    f'Service PID {proc.pid} exited {proc.returncode}; see node logs'
                )

    def wait(self, proc, timeout):
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError('Subprocess exceeded its configured time limit')
            time.sleep(2)
        if proc.returncode:
            raise RuntimeError(f'Subprocess exited {proc.returncode}; see logs')

    def healthy(self, url, timeout=3600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            try:
                with urllib.request.urlopen(url + '/health', timeout=5) as response:
                    value = json.loads(response.read())
                if value.get('loaded'):
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(f'Service did not become healthy: {url}')

    def close(self):
        for proc in reversed(self.children):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        for proc in self.children:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        for log in self.logs:
            log.close()


def serve_node(cfg, index, session):
    """Start this node's services, qualify render batch sizes, and write their identities.

    Returns `(processes, roles, urls)`. Writes `status.json`, `selected-batching.json` and
    `ready.json` in `<out>/node-<index>/`.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible == ['']:
        # The allocation holds the whole node; gpu_roles rejects fewer than eight GPUs.
        visible = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
        ).split()
    roles = gpu_roles(index, visible)
    root = Path(cfg['out'])
    out = root / f'node-{index}'
    processes = Processes(out, cfg)
    status = out / 'status.json'
    try:
        atomic_json(
            status, dict(state='starting', session=session, roles=roles, host=socket.gethostname())
        )
        processes.start(
            'gpu-utilization',
            [
                '/usr/bin/nvidia-smi',
                '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                '--format=csv',
                '-l',
                '5',
            ],
        )
        template = read_json(cfg['reward_config'])
        # The staging path depends on the node index, not the hostname, so the service identity
        # holds when a resumed allocation lands on other hosts.
        staging = str(out.resolve() / 'staging')
        Path(staging).mkdir(exist_ok=True)
        vqa_port = cfg['vqa_port']
        vqa_url = f'http://127.0.0.1:{vqa_port}'
        processes.start(
            'vqa',
            [
                cfg['vqa_python'],
                '-m',
                'axis_grpo.training.vqa_service',
                '--cache',
                cfg['cache'],
                '--staging',
                staging,
                '--port',
                str(vqa_port),
                '--model',
                template['vqa_model'],
                '--batch-size',
                str(cfg['vqa_batch']),
                '--probe-out',
                str(out / f'vqa-parity-{session}.json'),
                '--execute',
            ],
            [roles['vqa']],
            True,
        )
        processes.healthy(vqa_url)
        template.update(staging=staging, vqa_url=vqa_url)
        probe_config = out / 'probe-config.json'
        atomic_json(probe_config, template)
        probe_out = out / f'probe-{session}'
        probe = processes.start(
            'probe',
            [
                cfg['reward_python'],
                '-m',
                'axis_grpo.training.gpu_probe',
                '--config',
                str(probe_config),
                '--out',
                str(probe_out),
                '--execute',
            ],
            [roles['renderers'][0]],
        )
        processes.wait(probe, cfg['probe_timeout'])
        measured = read_json(probe_out / 'summary.json')
        if measured['status'] != 'passed':
            raise RuntimeError('T2I model/scorer qualification failed')
        selected = measured['selected_render_batch']
        # The first launch freezes the selected batch sizes; a resumed allocation must support
        # them, since a different batch size could change rendered images.
        selection = out / 'selected-batching.json'
        if selection.exists():
            prior = read_json(selection)
            selected = prior['render_batch']
            usable = {b['batch_size'] for b in measured['benchmark'] if 'images_per_second' in b}
            if selected not in usable:
                raise RuntimeError('Previous render batch no longer fits this allocation')
            val_selected = prior['val_render_batch']
            val_usable = {
                b['batch_size']
                for b in measured['validation_benchmark']
                if 'images_per_second' in b
            }
            if val_selected not in val_usable:
                raise RuntimeError('Previous validation batch no longer fits')
        else:
            val_selected = measured['selected_val_render_batch']
            atomic_json(
                selection,
                dict(
                    render_batch=selected,
                    val_render_batch=val_selected,
                    qualification=str(probe_out),
                ),
            )
        template['render_batch'] = selected
        template['val_render_batch'] = val_selected
        urls, service_hashes = [], {}
        for i, gpu in enumerate(roles['renderers']):
            config = out / f'reward-{i}.json'
            atomic_json(config, template)
            port = cfg['reward_port'] + i
            processes.start(
                f'reward-{i}',
                [
                    cfg['reward_python'],
                    '-m',
                    'axis_grpo.training.service',
                    '--config',
                    str(config),
                    '--out',
                    str(out / f'reward-{i}'),
                    '--host',
                    '0.0.0.0',
                    '--port',
                    str(port),
                    '--execute',
                ],
                [gpu],
                True,
            )
            urls.append(f'http://{socket.gethostname()}:{port}')
        for url in urls:
            service_hashes[url] = processes.healthy(url)['service_hash']
        atomic_json(
            out / 'ready.json',
            dict(
                session=session,
                host=socket.gethostname(),
                roles=roles,
                endpoints=urls,
                service_hashes=service_hashes,
                render_batch=selected,
            ),
        )
        atomic_json(
            status, dict(state='ready', session=session, roles=roles, host=socket.gethostname())
        )
        return processes, roles, urls
    except BaseException as error:
        atomic_json(
            status, dict(state='failed', session=session, error=f'{type(error).__name__}: {error}')
        )
        processes.close()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--node-index', type=int, default=0)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    cfg = read_json(a.config)
    if not a.execute:
        print(
            json.dumps(
                dict(
                    execute=False,
                    config=cfg,
                    layout=[gpu_roles(i, list(map(str, range(8)))) for i in (0, 1)],
                ),
                indent=2,
            )
        )
        return
    session = scheduler_job_id() or f'local-{os.getpid()}'
    os.environ['T_DIG_JOB_ID'] = session
    root = Path(cfg['out'])
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'status.json'
    children = Processes(root, cfg)
    local, remotes = None, []

    def stop(signum, frame):
        raise SystemExit(f'Received signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        nodes = cfg.get('nodes', 2)
        if a.node_index >= 1:
            local, _, _ = serve_node(cfg, a.node_index, session)
            while True:
                local.check()
                done = root / 'fleet-stop.json'
                if done.exists() and read_json(done).get('session') == session:
                    break
                time.sleep(2)
            return
        hosts = allocation_hosts(os.environ.get('T_DIG_NODEFILE'), nodes)
        if hosts[0].split('.')[0] != socket.gethostname().split('.')[0]:
            raise ValueError('Supervisor must start on the first allocated node')
        atomic_json(status, dict(state='qualifying', session=session, hosts=hosts))
        # Config contents never pass through the remote command line: the node reads the config
        # file. The launch command must stay in the foreground while the remote node runs.
        remotes = [
            children.start(
                f'remote-node-{i}',
                remote_command(cfg, a.config, session, i, hosts[i]),
            )
            for i in range(1, nodes)
        ]

        def remotes_alive():
            for i, remote in enumerate(remotes, 1):
                if remote.poll() is not None:
                    raise RuntimeError(f'Remote reward node {i} exited')

        local, roles, urls = serve_node(cfg, 0, session)
        deadline = time.monotonic() + cfg['probe_timeout'] + 3600
        pending = {i: root / f'node-{i}/ready.json' for i in range(1, nodes)}
        while pending:
            local.check()
            remotes_alive()
            for i, ready in list(pending.items()):
                value = read_json(ready) if ready.exists() else {}
                if value.get('session') == session:
                    urls += value['endpoints']
                    del pending[i]
            if time.monotonic() > deadline:
                raise TimeoutError('Remote reward node startup timed out')
            time.sleep(2)
        expected = 5 + 7 * (nodes - 1)
        if len(urls) != expected or len(set(urls)) != expected:
            raise ValueError('Incomplete reward fleet')
        config = read_json(cfg['train_config'])
        config['reward_endpoints'] = urls
        path = root / 'resolved-train-input.json'
        atomic_json(path, config)

        def wait_policy(proc):
            while proc.poll() is None:
                local.check()
                remotes_alive()
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f'Training exited {proc.returncode}; see train logs')

        # Before the full epoch, each job trains two steps on three real rows: it stops after the
        # first and resumes from that checkpoint in a fresh process.
        qualification = root / 'qualification-passed.json'
        if not qualification.exists():
            atomic_json(status, dict(state='qualifying_policy', session=session, hosts=hosts))
            smoke = read_json(cfg['qualification_config'])
            smoke['reward_endpoints'] = urls
            smoke_path = root / 'resolved-qualification-input.json'
            atomic_json(smoke_path, smoke)
            smoke_out = root / 'qualification-policy'
            if (smoke_out / 'manifest.json').exists():
                raise ValueError(
                    'Incomplete earlier qualification; inspect it before reusing this run directory'
                )
            command = [
                cfg['train_python'],
                '-m',
                'axis_grpo.training.run',
                '--config',
                str(smoke_path),
                '--out',
                str(smoke_out),
                '--execute',
            ]
            start = time.monotonic()
            wait_policy(
                children.start(
                    'qualification-first', command + ['--stop-after-step', '1'], roles['trainer']
                )
            )
            wait_policy(
                children.start('qualification-resume', command + ['--resume'], roles['trainer'])
            )
            steps = [read_json(f) for f in sorted((smoke_out / 'checkpoints/steps').glob('*.json'))]
            norms = [s['metrics'].get('actor/grad_norm', float('nan')) for s in steps]
            if (
                [i for s in steps for i in s['indices']] != [0, 1, 2]
                or not all(math.isfinite(n) for n in norms)
                or not any(n > 0 for n in norms)
            ):
                raise RuntimeError('Qualification failed gradient/curriculum checks')
            atomic_json(
                qualification,
                dict(
                    passed=True,
                    session=session,
                    seconds=time.monotonic() - start,
                    indices=[0, 1, 2],
                    gradient_norms=norms,
                    checkpoint_resume=True,
                ),
            )
        command = [
            cfg['train_python'],
            '-m',
            'axis_grpo.training.run',
            '--config',
            str(path),
            '--out',
            str(root / 'policy'),
            '--execute',
        ]
        if (root / 'policy/manifest.json').exists():
            command.append('--resume')
        atomic_json(
            status,
            dict(
                state='training',
                session=session,
                hosts=hosts,
                policy_gpus=2,
                renderer_gpus=len(urls),
                fidelity_gpus=nodes,
                reward_endpoints=urls,
            ),
        )
        trainer = children.start('train', command, roles['trainer'])
        wait_policy(trainer)
        atomic_json(status, dict(state='complete', session=session, hosts=hosts))
    except BaseException as error:
        if a.node_index == 0:
            atomic_json(
                status,
                dict(state='failed', session=session, error=f'{type(error).__name__}: {error}'),
            )
        raise
    finally:
        if a.node_index == 0:
            atomic_json(root / 'fleet-stop.json', dict(session=session))
            # Let the remote launchers see the stop file and close their services before
            # their launch commands are killed.
            deadline = time.monotonic() + 60
            while any(r.poll() is None for r in remotes) and time.monotonic() < deadline:
                time.sleep(2)
        if local:
            local.close()
        children.close()


if __name__ == '__main__':
    main()
