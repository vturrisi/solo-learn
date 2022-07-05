import getpass
import os
import re
import socket
from argparse import ArgumentParser
from contextlib import closing

import paramiko
import psutil
import torch
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--hostfile", type=str, default="/horovod/generated/hostfile")
parser.add_argument("--bash_file", type=str, required=True)
parser.add_argument("--github_user", type=str, default=None)
parser.add_argument("--github_repo", type=str, default=None)
parser.add_argument("--github_token", type=str, default=None)
parser.add_argument("--extra_script_args", default=[], type=str, nargs="+")

args = parser.parse_args()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def connect_and_execute(host, commands):
    if not isinstance(commands, list):
        commands = [commands]
    ssh.connect(host)
    for command in commands:
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)

        ssh_stdout.channel.recv_exit_status()
    ssh.close()


def parse_python_bash(folder, file):
    # parse python command
    os.chdir(folder)
    with open(file) as f:
        python_command = re.sub(
            " +",
            " ",
            "".join(f.readlines())
            .replace("\n", "")
            .replace("\\", "")
            .replace("python3", "")
            .strip(),
        )
        gpus_str = ",".join((str(i) for i in range(gpus_per_node)))
        # fix number of devices
        python_command = re.sub("--devices [^-]* -", f"--devices {gpus_str} -", python_command)
        # add arguments to command
        for i, arg in enumerate(args.extra_script_args, start=1):
            python_command = python_command.replace(f"${i}", arg)
    os.chdir("..")


# gather hosts
hosts = []
with open(args.hostfile) as f:
    for line in f:
        host, *_ = line.split()
        hosts.append(host)

num_nodes = len(hosts)
gpus_per_node = torch.cuda.device_count()


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# clone repo in all hosts
if args.github_repo is not None:
    assert args.github_user is not None

    if args.github_token is None:
        commands = [
            f"rm -rf {args.github_repo}",
            f"git clone https://github.com/{args.github_user}/{args.github_repo}.git",
        ]
    else:
        commands = [
            f"rm -rf {args.github_repo}",
            f"git clone https://{args.github_user}:{args.github_token}@github.com/{args.github_user}/{args.github_repo}.git",
        ]

    for host in tqdm(hosts, desc="Cloning repository into nodes"):
        connect_and_execute(host, commands)

# parse python command
python_command = parse_python_bash(args.github_repo, args.bash_file)

# find a free port for distributed
PORT = find_free_port()

# execute command
channels = []
for i, host in enumerate(tqdm(hosts, desc="Calling distributed_training")):
    # to run with torch.distributed.run (just as backup)
    # command = re.sub(
    #     " +",
    #     " ",
    #     f"""cd {args.github_repo};python3 -m torch.distributed.run \
    #     --nnodes={num_nodes} \
    #     --nproc_per_node={gpus_per_node} \
    #     --master_addr {hosts[0]} \
    #     --master_port {PORT} \
    #     --node_rank {i} \
    #     {python_command} \
    #     --num_nodes {num_nodes}""",
    # )

    command = re.sub(
        " +",
        " ",
        f"""
        cd {args.github_repo}; \
        MASTER_ADDR={hosts[0]} \
        MASTER_PORT={PORT} \
        WORLD_SIZE={num_nodes} \
        NODE_RANK={i} \
        python3 {python_command} \
        --num_nodes {num_nodes}""",
    )

    ssh.connect(host)
    channel = ssh.get_transport().open_session()
    channels.append(channel)
    channel.exec_command(command)

print("Awaiting for processes to end")
for channel in channels:
    channel.recv_exit_status()
    channel.close()


print("Killing all leftovers")
user_name = getpass.getuser()
pids = {
    proc.pid
    for proc in psutil.process_iter()
    if proc.username() == user_name and "python3" in proc.name()
}
cur_pid = os.getpid()
for pid in pids:
    if pid != cur_pid:
        psutil.Process(pid).terminate()
