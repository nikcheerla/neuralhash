
import random, sys, os, glob, yaml
import argparse, subprocess, shutil
from fire import Fire


def run(cmd, type='experiment', config="default", shutdown=True):

	try:
		run_log = yaml.load(open("jobs/runlog.yml"))
	except:
		run_log = {}

	run_data = run_log[type] = run_log.get(type, {})
	run_data["runs"] = run_data.get("runs", 0) + 1
	run_name = args.type + str(run_data["runs"])
	run_data = run_data.get(run_name, {"config": config, "cmd": cmd})

	process = subprocess.Popen(cmd, shell=True, start_new_session=True, \
							stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	for line in iter(process.stdout.readline):
		print (line)

	while process.poll() is None:
		status = ""
		try:
			pass
		except KeyboardInterrupt:
			run_data['status'] = "Killed"
			process.kill()

	run_data["status"] = "Complete" if process.poll() == 0 else "Shutdown"
		
	shutil.copytree("output", f"jobs/{run_name}/output")
	shutil.rmtree("output/")
	os.makedirs("output/")

	yaml.safe_dump(run_log, open("jobs/runlog.yml", 'w'), \
			allow_unicode=True, default_flow_style=False)
	yaml.safe_dump(run_data[run_name], open(f"jobs/{run_name}/comments.yml", 'w'), \
			allow_unicode=True, default_flow_style=False)

	if args.shutdown and not keyboard_interrupt:
		subprocess.call("sudo shutdown -h now", shell=True)



if __name__ == "__main__":
	Fire(run)


