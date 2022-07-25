from goprocam import GoProCamera, constants, exceptions
from datetime import datetime

interface = "en7"

gp = GoProCamera.GoPro(ip_address=GoProCamera.GoPro.getWebcamIP(interface), 
    camera=constants.gpcontrol, 
    webcam_device=interface,api_type=constants.ApiServerType.OPENGOPRO)

try:
    r = gp.setWiredControl(constants.on)
    print(r)
except exceptions.WiredControlAlreadyEstablished:
    pass  # sometimes throws 500 server error when camera is already on wired control mode

# start_time = datetime.now()
ttf=0
for i in range(10):
    start_time = datetime.now()
    gp.take_photo()
    gp.downloadLastMedia()#custom_filename="gp/gopro_most_recent.jpg")
    ttf += (datetime.now() - start_time).total_seconds()

# dt = (datetime.now() - start_time).total_seconds()
# print('FPS 10 frames: ', str(10/dt))
print('mean seconds to frame: ', str(ttf/10))