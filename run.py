

# FIX RUN.PY

import random, sys, os, glob, yaml
import argparse, subprocess, shutil

parser = argparse.ArgumentParser(description='Runs arbitary jobs with experiment configging and output saving.')

parser.add_argument("cmd")
parser.add_argument('--type', default="experiment", help='type of run')
parser.add_argument('--config', default="configuration", help='type of run')
parser.add_argument('--shutdown', action='store_const', const=True, default=False, 
	help='shutdown instance when finished')
args = parser.parse_args()

try:
	run_log = yaml.load(open("jobs/runlog.yml"))
except:
	run_log = {}

run_data = run_log[args.type] = run_log.get(args.type, {})
run_data["runs"] = run_data.get("runs", 0) + 1
run_name = args.type + str(run_data["runs"])

run_data[run_name] = {"config": args.config, "cmd": args.cmd, "status": "In Progress"}


process = subprocess.Popen(args.cmd, shell=True)
keyboard_interrupt = False
def monitor_process(process, run_data):

	while process.poll() is None:
		try:
			pass
		except KeyboardInterrupt:
			# Program shut down
			run_data[run_name] = {"config": args.config, "cmd": args.cmd, "status": "Killed"}
			process.kill()
			keyboard_interrupt = True
			return
		except SystemExit:
			# Program shut down
			run_data[run_name] = {"config": args.config, "cmd": args.cmd, "status": "Shutdown"}
			process.kill()
			return

	if process.poll() == 0:
		result = "result omitted for clarity" #input ("Add a comment describing results? [ENTER to skip]: ")
		run_data[run_name] = {"config": args.config, "cmd": args.cmd, \
							"status": "Complete", "results": result}
	else:
		run_data[run_name] = {"config": args.config, "cmd": args.cmd, "status": "Error"}

monitor_process(process, run_data)

shutil.copytree("output", f"jobs/{run_name}/output")
shutil.rmtree("output/")
os.makedirs("output/")

yaml.safe_dump(run_log, open("jobs/runlog.yml", 'w'), \
		allow_unicode=True, default_flow_style=False)
yaml.safe_dump(run_data[run_name], open(f"jobs/{run_name}/comments.yml", 'w'), \
		allow_unicode=True, default_flow_style=False)

if args.shutdown and not keyboard_interrupt:
	subprocess.call("sudo shutdown -h now", shell=True)




