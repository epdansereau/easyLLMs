import os
import time
import pprint

class ChunkDebugger:
    '''
    from debug import chunkdebug as cc
    cc.init() # creates a file in 'logs/timestamp', where timestamp is the current timestamp
    for chunk in some operator:
        yield cc(chunk) # will pretty print chunk in 'logs/timestamp' with pprint
    '''
    
    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, self.timestamp)
        self.file_handle = open(self.log_file, "w", encoding="utf-8")
    
    def init(self):
        """Reinitialize the debugger with a new log file."""
        if self.file_handle:
            self.file_handle.close()
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, self.timestamp)
        self.file_handle = open(self.log_file, "w", encoding="utf-8")
    
    def __call__(self, chunk):
        """Log the chunk and return it for further processing."""
        pprint.pprint(chunk, stream=self.file_handle)
        self.file_handle.flush()
        return chunk
    
    def __del__(self):
        if self.file_handle:
            self.file_handle.close()

# Create a global instance of ChunkDebugger
chunkdebug = ChunkDebugger()
