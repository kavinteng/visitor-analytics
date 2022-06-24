import os
import datetime

run_check = False
while True:
    time = datetime.datetime.now().strftime("%T")
    if time > '08:30:00' and run_check == False:
        os.system('python main4.py')
        run_check = True

    elif time < '08:10:00' and time > '08:00:00' and run_check == True:
        run_check = False