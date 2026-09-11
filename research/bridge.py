"""CPU-only S3 artifact bridge; credentials stay inside the Iris task."""

import time

if __name__ == "__main__":
    print("RESEARCH_BRIDGE_READY", flush=True)
    while True:
        time.sleep(30)
