import argparse
import warnings
import os.path
import re
import subprocess

CONFIG_DIR = "/Users/Antonio/idtrackerai/config"
DATA_DIR = "/Dropbox/FlySleepLab_Dropbox/Data/flyhostel_data/videos"

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--datetime", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--environment", default="idtrackerai4")
    ap.add_argument("--knowledge-transfer", dest="knowledge_transfer", default=None)
    ap.add_argument("--interval", nargs="+", type=int,required=True)
    return ap

def build_idtrackerai_call(experiment_folder, chunk, config_file):
    idtrackerai_call = "idtrackerai terminal_mode"
    idtrackerai_call += f" --_session {chunk}"
    idtrackerai_call += f" --_video {experiment_folder}/{chunk}.avi"
    idtrackerai_call += f" --load  {config_file}"
    idtrackerai_call += " --exec track_video"
    print(idtrackerai_call)

    return idtrackerai_call


def write_jobfile(idtrackerai_call, jobfile, chunk="000000", environment="idtrackerai", knowledge_transfer=None):

    lines = [
        "#! /bin/bash",
        f"source ~/.bashrc_conda && conda activate {environment}",
    ]

    datetime_str = os.path.basename(os.path.dirname(jobfile.strip("/")).strip("/"))


    if knowledge_transfer:

        if knowledge_transfer == "previous":
            lines.append('echo "from idtrackerai.utils.idtrackerai_loop import setup_knowledge_transfer" > local_settings.py')
            function_call = f'setup_knowledge_transfer("{datetime_str}",{int(chunk)-1})'
            lines.append(f"echo 'IDENTITY_TRANSFER, KNOWLEDGE_TRANSFER_FOLDER_IDCNN={function_call}' >> local_settings.py")
        else:
            lines.append('echo IDENTITY_TRANSFER=True > local_settings.py')
            lines.append(f'echo KNOWLEDGE_TRANSFER_FOLDER_IDCNN=\\"{knowledge_transfer}\\" >> local_settings.py')

        local_settings_py_backup = os.path.join(os.path.dirname(jobfile), f"session_{chunk}-local_settings.py")
        lines.append(f"cp local_settings.py {local_settings_py_backup}")

    lines.append(idtrackerai_call)

    with open(jobfile, "w") as fh:
        for line in lines:
            fh.write(f"{line}\n")

    return


def build_qsub_call(experiment_folder, chunk, config_file, **kwargs):

    # prepare the call to idtrackerai
    idtrackerai_call = build_idtrackerai_call(experiment_folder, chunk, config_file)

    # save the call together with a setup block into a script
    jobfile = os.path.join(experiment_folder, f"session_{chunk}.sh")
    write_jobfile(idtrackerai_call, jobfile, chunk=chunk, **kwargs)

    # add the qsub flags
    output_file = os.path.join(experiment_folder, f"session_{chunk}_output.txt")
    error_file = os.path.join(experiment_folder, f"session_{chunk}_error.txt")
    job_name = f"session_{chunk}"
    cmd = f"qsub -o {output_file} -e {error_file} -N {job_name} {jobfile}"
    return cmd.split(" ")


def run_one_loop(experiment_folder, chunk, config_file, **kwargs):

    qsub_call = build_qsub_call(experiment_folder, chunk, config_file, **kwargs)
    process = subprocess.Popen(qsub_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr



def setup_knowledge_transfer(*args, **kwargs):

    network_folder = get_network_folder(*args, **kwargs)
    if network_folder is None:
        return False, None
    else:
        return True, network_folder

def get_network_folder(datetime_str, i):

    if i < 0:
        return None

    session_folder = os.path.join(DATA_DIR, datetime_str, f"session_{str(i).zfill(6)}")

    if not os.path.exists(session_folder):
        warnings.warn(f"{session_folder} does not exist")
        return None

    folders_in_session = os.listdir(session_folder)
    accum_folders = []
    for folder in folders_in_session:
        if re.search("accumulation_[0-9]", folder):
            accum_folders.append(os.path.join(session_folder, folder))

    if len(accum_folders) == 0:
        return get_network_folder(datetime_str, i-1)
    else:
        last_network = sorted(accum_folders)[-1]
        return last_network


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    if args.side == "left":
        side_code = "_1"
    elif args.side == "right":
        side_code = "_2"
    else:
        side_code = ""

    config_file = os.path.join(CONFIG_DIR, args.datetime + side_code + ".conf")
    experiment_folder = os.path.join(DATA_DIR, args.datetime)

    for i in range(*args.interval, 1):
        chunk = str(i).zfill(6)
        run_one_loop(
            experiment_folder, chunk, config_file,
            environment=args.environment,
            knowledge_transfer=args.knowledge_transfer
        )


def loop(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == "__main__":
    main()
