
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil
from fire import Fire
from utils import elapsed

import IPython


def run(cmd, mode='experiment', config="default", shutdown=False, ignore_error=False):

	elapsed()
	try:
		run_log = yaml.load(open("jobs/runlog.yml"))
	except:
		run_log = {}

	run_data = run_log[mode] = run_log.get(mode, {})
	run_data["runs"] = run_data.get("runs", 0) + 1
	run_name = mode + str(run_data["runs"])
	run_data[run_name] = run_data.get(run_name, 
				{"config": config, "cmd": cmd, "status": "Running"})
	run_data = run_data[run_name]

	print (f"Running job: {run_name}")
	os.makedirs(f"jobs/{run_name}", exist_ok=True)

	cmd = cmd.split()
	if cmd[0] =='python': cmd.insert(1, '-u')
	cmd = " ".join(cmd) + f" | tee jobs/{run_name}/stdout.txt"
	process = subprocess.Popen(cmd.split(), shell=False, \
							universal_newlines=True)

	try:
		return_code = process.wait()
		run_data["status"] = "Error" if return_code else "Complete"
	except KeyboardInterrupt:
		print ("\nKilled by user.")
		process.kill()
		run_data["status"] = "Killed"

	process.kill()

	if ignore_error and run_data["status"] != "Complete":
		return
		
	shutil.copytree("output", f"jobs/{run_name}/output")
	shutil.rmtree("output/")
	os.makedirs("output/")

	yaml.safe_dump(run_log, open("jobs/runlog.yml", 'w'), \
			allow_unicode=True, default_flow_style=False)
	yaml.safe_dump(run_data, open(f"jobs/{run_name}/comments.yml", 'w'), \
			allow_unicode=True, default_flow_style=False)

	interval = elapsed()
	print (f"Program ended after {interval:0.4f} seconds.")
	if shutdown and run_data["status"] != "Killed" and interval > 60:
		print (f"Shutting down in 1 minute.")
		time.sleep(60)
		subprocess.call("sudo shupdown -h now", shell=True)



if __name__ == "__main__":
	Fire(run)


