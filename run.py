
import random, sys, os, glob
import argparse, subprocess

parser = argparse.ArgumentParser(description='Runs arbitary jobs with experiment tagging and output saving.')

parser.add_argument("cmd")
parser.add_argument('--type', default="experiment", help='type of run')
parser.add_argument('--tag', default="experiment", help='tag for the experiment')
args = parser.parse_args()

run_log = yaml.load(open(".runlog.yml"))
run_data = run_log[args.type] = run_log.get(args.type, {})
run_data["runs"] = run_data.get("runs", 0) + 1
run_name = args.type + str(run_data["runs"])

run_data[run_name] = {"tag": args.tag, "status": "In Progress"}

process = subprocess.Popen(args.cmd, shell=True)

def monitor_process(process, run_data):

	while process.poll() is None:
		try:
			pass
		except (KeyboardInterrupt, SystemExit):
			# Program shut down
			run_data[run_name] = {"tag": args.tag, "status": "Shutdown"}
			return

	try:
	 	subprocess.check_call(command)
		comment = input("Add a comment describing results? [ENTER to skip]: ")
		run_data[run_name] = {"tag": args.tag, "status": "Complete", "comment": comment}

	except subprocess.CalledProcessError:
		run_data[run_name] = {"tag": args.tag, "status": "Error"}

monitor_process(process, run_data)
yaml.dump(run_log, ".runlog.yml")

output_directory = "outputs/"
yaml.dump(run_log, ".yml")





