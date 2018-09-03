
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil, shlex
from fire import Fire
from utils import elapsed

import IPython


def execute(cmd, mode="experiment", config="default", shutdown=False, debug=False):

    elapsed()
    try:
        run_log = yaml.load(open("jobs/runlog.yml"))
    except:
        run_log = {}

    run_data = run_log[mode] = run_log.get(mode, {})
    run_data["runs"] = run_data.get("runs", 0) + 1
    run_name = mode + str(run_data["runs"])
    run_data[run_name] = run_data.get(run_name, {"config": config, "cmd": cmd, "status": "Running"})
    run_data = run_data[run_name]

    print(f"Running job: {run_name}")

    shutil.rmtree("output/", ignore_errors=True)
    os.makedirs("output/")
    os.makedirs(f"jobs/{run_name}", exist_ok=True)

    with open("jobs/jobinfo.txt", "w") as config_file:
        print(config, file=config_file)

    cmd = shlex.split(cmd)
    if cmd[0] == "python" and debug:
        cmd[0] = "ipython"
        cmd.insert(1, "-i")
    elif cmd[0] == "python":
        cmd.insert(1, "-u")

    print(" ".join(cmd))
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, universal_newlines=True)

    try:
        with open(f"jobs/{run_name}/stdout.txt", "w") as outfile:
            for stdout_line in iter(process.stdout.readline, ""):
                print(stdout_line, end="")
                outfile.write(stdout_line)

        return_code = process.wait()
        run_data["status"] = "Error" if return_code else "Complete"
    except KeyboardInterrupt:
        print("\nKilled by user.")
        process.kill()
        run_data["status"] = "Killed"
    except OSError:
        print("\nSystem error.")
        process.kill()
        run_data["status"] = "Error"

    process.kill()

    if debug and run_data["status"] != "Complete":
        return

    shutil.copytree("output", f"jobs/{run_name}/output")
    for file in glob.glob("*.py"):
        shutil.copy(file, f"jobs/{run_name}")

    yaml.safe_dump(run_log, open("jobs/runlog.yml", "w"), allow_unicode=True, default_flow_style=False)
    yaml.safe_dump(run_data, open(f"jobs/{run_name}/comments.yml", "w"), allow_unicode=True, default_flow_style=False)

    interval = elapsed()
    print(f"Program ended after {interval:0.4f} seconds.")
    if shutdown and run_data["status"] != "Killed" and interval > 60:
        print(f"Shutting down in 1 minute.")
        time.sleep(60)
        subprocess.call("sudo shutdown -h now", shell=True)


def run(cmd, mode="experiment", config="default", shutdown=False, debug=False):
    cmd = f""" screen -S {config} bash -c "python run.py execute \\"{cmd}\\" --mode {mode} --config {config} --shutdown {shutdown} --debug {debug}" """
    subprocess.call(shlex.split(cmd))


if __name__ == "__main__":
    Fire({"run": run, "execute": execute})
