import config
from datetime import datetime

fixed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' '

if config.door_status():
    config.set_status(True)
    print("Esik ashyq tur")
else:
    config.set_status(False)
    print("Esik ashyldy")

print(fixed_time)