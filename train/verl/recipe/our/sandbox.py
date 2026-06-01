from flask import Flask, request, jsonify
import sys
import io
import contextlib
import traceback
import multiprocessing
import time

app = Flask(__name__)


def execute_unsafe_code(code, timeout_sec, result_queue):
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    exec_globals = {}

    result = {
        "status": "success",
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "run_status": "Finished"
    }

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, exec_globals)
    except Exception:
        result["status"] = "failed"
        result["run_status"] = "Error"
        result["return_code"] = 1
        stderr_capture.write(traceback.format_exc())

    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

    result_queue.put(result)


@app.route('/run_code', methods=['POST'])
def run_code():
    try:
        data = request.json
        code = data.get('code', '')
        if isinstance(code, list):
            code = code[0]

        timeout_sec = int(data.get('timeout', 10))

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=execute_unsafe_code, args=(code, timeout_sec, queue))
        p.start()
        p.join(timeout_sec)

        if p.is_alive():
            p.terminate()
            p.join()
            return jsonify({
                "status": "timeout",
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": "",
                    "stderr": f"Error: Execution timed out after {timeout_sec} seconds.",
                    "return_code": 1
                },
                "error": "Execution timed out."
            })

        if queue.empty():
            return jsonify({
                "status": "failed",
                "run_result": {
                    "status": "Error",
                    "stdout": "",
                    "stderr": "Error: Process crashed or produced no output.",
                    "return_code": 1
                },
                "error": "Process crashed"
            })

        result = queue.get()

        return jsonify({
            "status": result["status"],
            "run_result": {
                "status": result["run_status"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "return_code": result["return_code"]
            },
            "error": result["stderr"] if result["status"] != "success" else None
        })

    except Exception as e:
        print(f"Server Error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return "OK", 200


if __name__ == '__main__':
    print("Starting Sandbox Server on port 8080 (Threaded Mode)...")
    app.run(host='0.0.0.0', port=8080, threaded=True)