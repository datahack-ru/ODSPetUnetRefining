import time
import logging

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, name: str, maxCount = None):
        self.name = name
        self.time = 0
        self.count = 0
        self.maxCount = maxCount

    def start(self):
        self.startTime = time.time()
        if self.maxCount is not None:
            self.time = 0
            self.count = 0
    
    def end(self):
        self.time += time.time() - self.startTime
        self.count += 1
    
    def append(self):
        self.time += time.time() - self.startTime
    
    def after_append(self):
        self.count += 1
    
    def averageTime(self):
        return self.time / self.count if self.count > 0 else -1
    
    def printAverageTime(self):
        print('{} time {}'.format(self.name, self.averageTime()))

    def logAverageTime(self):
        logger.debug('%s time %s', str(self.name), str(self.averageTime()))
