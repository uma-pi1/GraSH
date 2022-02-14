def monitor_gpus(folder, interval=1):
    import os
    import time
    import logging
    import pynvml
    import psutil

    try:
        pynvml.nvmlInit()
    except Exception:
        print("could not initialize GPU monitor")
        return
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        return
    logger = logging.getLogger("gpu_monitor")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, f"gpu_monitor.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # write header
    #logger.info(
    #    f"timestamp; device_id; gpu_util; gpu_mem_util; power_usage; total_consumption; cpu_util; cpu_mem_util"
    #)
    while True:
        time.sleep(interval)
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
            res = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_res = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_consumption = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            cpu_util = psutil.cpu_percent()
            cpu_mem_util = psutil.virtual_memory().percent
            logger.info(
                f"{time.time()}; {i}; {res.gpu}; " +
                f"{round((mem_res.used/mem_res.total)*100)}; " +
                f"{power_usage}; {total_consumption}; {cpu_util}; {cpu_mem_util}"
            )