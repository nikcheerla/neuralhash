import sys
from subprocess import call

try:
    call(['sleep', '10'], start_new_session=True)
except KeyboardInterrupt:
    print('Ctrl C')
else:
    print('no exception')