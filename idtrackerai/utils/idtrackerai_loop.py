import argparse
import warnings
import os
import os.path
import re
import subprocess
import json

FORBIDDEN_FIELDS = ["session", "_chunk", "_video"]
ANALYSIS_FOLDER_NAME="idtrackerai"

def get_analysis_folder(folder):
    return os.path.join(folder, ANALYSIS_FOLDER_NAME)

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help=
        "Name of imgstore repository or folder with multiple videos. If the input has structure X/./Y is passed in it,"
        " all paths will be relative to X (and not absolute i.e. relative to /)",
    )
    ap.add_argument(
        "--suffix",
        required=False,
        help="""Suffix on the name of the config file (before .conf extension).
    This is useful if more than one analysis is to be done on the same video""",
    )
    ap.add_argument(
        "--environment",
        default=None,
        help="Conda environment to use when running idtrackerai",
    )
    ap.add_argument(
        "--knowledge-transfer",
        dest="knowledge_transfer",
        default=None,
        help="""Whether to enable knowledge transfer during the analysis of consecutive chunks.
        If you want to enable this, please pass previous (to enable transfer from the previous chunk).""",
    )
    ap.add_argument(
        "--interval",
        nargs="+",
        type=int,
        required=True,
        help="Chunks to analyze from first to last (last does not count). Example to analyze 0-10 pass 0 11",
    )
    return ap


def check_forbidden_fields(config, fields):
    for field in fields:
        if field in config:
            raise Exception(f"Please remove {field} from the config file and run again")




def build_idtrackerai_call(experiment_folder, chunk, config_file, api="imgstore"):
    idtrackerai_call = "idtrackerai terminal_mode"
    idtrackerai_call += f" --_session {chunk}"

    assert api in ["imgstore", "cv2"]

    if api=="imgstore":
        idtrackerai_call += f" --_imgstore {experiment_folder}/metadata.yaml --_chunk {chunk}"
    elif api=="cv2":
        idtrackerai_call += f" --_video {experiment_folder}/idtrackerai/{chunk}.avi"
    idtrackerai_call += f" --load  {config_file}"
    idtrackerai_call += " --exec track_video"
    print(idtrackerai_call)

    return idtrackerai_call


def write_jobfile(
    lines,
    idtrackerai_call,
    experiment_folder,
    jobfile,
    chunk="000000",
    environment="idtrackerai",
    knowledge_transfer=None,
):
    analysis_folder = get_analysis_folder(experiment_folder)
    

    if environment is not None:
        lines.append(f"source ~/.bashrc_conda && conda activate {environment}")

    lines.append(f"mkdir -p {analysis_folder}")

    # this is the first line of the local_settings.py!
    # please make sure it has the > character once
    # all other appends to local_settings.py should have >>

    lines.append("echo SETTINGS_PRIORITY=1 > local_settings.py")

    lines.append(
        f"echo NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS=-2 >> local_settings.py"
    )

    if knowledge_transfer:

        if knowledge_transfer == "previous":
            lines.append(
                'echo "from idtrackerai.utils.idtrackerai_loop import setup_knowledge_transfer" >> local_settings.py'
            )
            function_call = f'setup_knowledge_transfer(experiment_folder="{analysis_folder}", i={int(chunk)-1})'
            lines.append(
                f"echo 'IDENTITY_TRANSFER, KNOWLEDGE_TRANSFER_FOLDER_IDCNN={function_call}' >> local_settings.py"
            )
        else:
            lines.append("echo IDENTITY_TRANSFER=True > local_settings.py")
            lines.append(
                f'echo KNOWLEDGE_TRANSFER_FOLDER_IDCNN=\\"{knowledge_transfer}\\" >> local_settings.py'
            )

        lines.append("echo 'print(IDENTITY_TRANSFER)' >> local_settings.py")
        lines.append(
            "echo 'print(KNOWLEDGE_TRANSFER_FOLDER_IDCNN)' >> local_settings.py"
        )

        local_settings_py_backup = os.path.join(
            experiment_folder, f"session_{chunk}-local_settings.py"
        )
        lines.append(f"cp local_settings.py {local_settings_py_backup}")

        lines.append(f"ln -s {experiment_folder}/{chunk}.avi {analysis_folder}/{chunk}.avi")

    lines.append(idtrackerai_call)

    with open(jobfile, "w") as fh:
        for line in lines:
            fh.write(f"{line}\n")

    return


def build_qsub_call(experiment_folder, chunk, config_file, **kwargs):

    # prepare the call to idtrackerai
    folder_split = experiment_folder.split("/./")
    if len(folder_split) == 1:
        working_directory = os.environ["HOME"]
        relative_experiment_folder = folder_split[0]
    else:
        working_directory = folder_split[0]
        relative_experiment_folder = folder_split[1]

    idtrackerai_call = build_idtrackerai_call(
        relative_experiment_folder, chunk, config_file.replace(working_directory, "").lstrip("/")
    )

    # save the call together with a setup block into a script
    analysis_folder = get_analysis_folder(experiment_folder)
    os.makedirs(analysis_folder, exist_ok=True)
    jobfile = os.path.join(analysis_folder, f"session_{chunk}.sh")
    lines = [
        "#! /bin/bash",
        f"cd {working_directory}",
    ]

    write_jobfile(lines, idtrackerai_call, relative_experiment_folder, jobfile, chunk=chunk, **kwargs)

    # add the qsub flags
    output_file = os.path.join(
        analysis_folder, f"session_{chunk}_output.txt"
    )
    error_file = os.path.join(analysis_folder, f"session_{chunk}_error.txt")
    job_name = f"session_{chunk}"
    cmd = f"qsub -o {output_file} -e {error_file} -N {job_name} -cwd {jobfile}"
    print(cmd)
    return cmd.split(" ")


def run_one_loop(experiment_folder, chunk, config_file, **kwargs):

    qsub_call = build_qsub_call(
        experiment_folder, chunk, config_file, **kwargs
    )
    process = subprocess.Popen(
        qsub_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return stdout, stderr


def setup_knowledge_transfer(*args, **kwargs):

    network_folder = get_network_folder(*args, **kwargs)
    if network_folder is None:
        return False, None
    else:
        return True, network_folder


def get_network_folder(experiment_folder, i):

    if i < 0:
        return None

    session_folder = os.path.join(
        experiment_folder, f"session_{str(i).zfill(6)}"
    )

    if not os.path.exists(session_folder):
        warnings.warn(f"{session_folder} does not exist")
        return None

    folders_in_session = os.listdir(session_folder)
    accum_folders = []
    for folder in folders_in_session:
        if re.search("accumulation_[0-9]", folder):
            accum_folders.append(os.path.join(session_folder, folder))

    if len(accum_folders) == 0:
        return get_network_folder(experiment_folder, i - 1)
    else:
        # last_network = sorted(accum_folders)[-1]
        # return last_network
        # TODO
        # Once this issue is solved, find the right way to select the best network
        # https://gitlab.com/polavieja_lab/idtrackerai/-/issues/65
        first_network = sorted(accum_folders)[0]
        assert os.path.exists(os.path.join(first_network, "model_params.npy"))
        return first_network


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    experiment_name = os.path.basename(args.input.rstrip("/"))
    if args.suffix != "":
        config_file = os.path.join(
            args.input, experiment_name + "_" + args.suffix + ".conf"
        )
    else:
        config_file = os.path.join(
            args.input, experiment_name + ".conf"
        )

    with open(config_file, "r") as filehandle:
        config = json.load(filehandle)
    check_forbidden_fields(config, FORBIDDEN_FIELDS)

    for i in range(*args.interval, 1):
        chunk = str(i).zfill(6)
        run_one_loop(
            args.input,
            chunk,
            config_file,
            environment=args.environment,
            knowledge_transfer=args.knowledge_transfer,
        )


def loop(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == "__main__":
    main()
