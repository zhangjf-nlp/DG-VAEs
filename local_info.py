# this file is just designed to describe which machine the experiemts are carried on
import platform
import uuid
from cuda_utils import get_device_name
machine_mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
machine_gpu = get_device_name()
machine_os = platform.platform() + ' ' + platform.architecture()[0]
machine_name = f"{machine_os} @ {machine_mac} - {machine_gpu}"