import sys, nbformat, time
from nbclient import NotebookClient
for path in sys.argv[1:]:
    t = time.time()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": "."}})
    try:
        client.execute()
        status = "OK"
    except Exception as e:
        status = f"FAIL: {str(e)[-3000:]}"
    nbformat.write(nb, path)
    print(path, status, f"{time.time()-t:.0f}s", flush=True)
    if status != "OK":
        sys.exit(1)
