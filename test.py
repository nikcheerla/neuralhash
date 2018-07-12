
import subprocess
try:

	proc = subprocess.Popen("ls -lrt", shell=True,\
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	for line in iter(proc.stdout.readline, ""):
		print (line.decode(), end="")

except KeyboardInterrupt:
	print ("Got Keyboard interrupt")