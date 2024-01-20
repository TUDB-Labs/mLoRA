import json
import shlex
import subprocess
import sys
import threading
import uuid

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/run/finetune', methods=['POST'])
def run():
    params = request.json
    config = params['config']
    process_id = str(uuid.uuid4())  # 生成一个唯一的进程ID
    file_name = process_id + '.json'
    configFile = open(file_name, "w")

    configFile.write(json.dumps(config, indent=4))
    print(params)
    shell_str = 'python ../mlora.py '

    shell_str += ' --base_model ' + params['base_model']
    shell_str += ' --model_type ' + params['model_type']
    if params['inference'] == 'true':
        shell_str += ' --inference'
    if params['load_lora'] == 'true':
        shell_str += ' --load_lora'
    if params['disable_lora'] == 'true':
        shell_str += ' --disable_lora'

    if params['tokenizer']:
        shell_str += ' --tokenizer ' + params['tokenizer']

    if params['load_8bit'] == 'true':
        shell_str += ' --load_8bit'
    if params['load_4bit'] == 'true':
        shell_str += ' --load_4bit'

    if params['device']:
        shell_str += ' --device ' + params['device']

    if params['config']:
        shell_str += ' --config ' + file_name

    if params['seed']:
        shell_str += ' --seed ' + str(params['seed'])

    if params['log'] == 'true':
        shell_str += ' --log True'

    process_id = str(uuid.uuid4())  # 生成一个唯一的进程ID
    print(shell_str)
    # 启动进程但不等待其完成
    execmd_thread = threading.Thread(target=execmd, args=(shell_str, process_id, True))
    execmd_thread.start()
    # 返回进程ID给客户端
    return jsonify({"process_id": process_id}), 200


@app.route('/getlog', methods=['GET'])
def get_log():
    process_id = request.args.get('process_id')
    process_info = processes.get(process_id)
    if process_info and 'output' in process_info:
        return json.dumps(process_info['output'])
    else:
        return '{"data": "Process not found or no output available"}'


@app.route('/stop/finetune', methods=['POST'])
def stop_finetune():
    """
    这个视图函数终止一个正在运行的进程。
    """
    process_id = request.json.get('process_id')
    process_info = processes.pop(process_id, None)
    if process_info:
        process_info['process'].terminate()  # 优雅地终止进程
        return jsonify({"message": "Process terminated"}), 200
    else:
        return jsonify({"error": "Process not found"}), 404


processes = {}


def execmd(command, process_id, shell=False):
    sp = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    processes[process_id] = {'process': sp, 'output': [], 'finished': False}  # 添加'finished'标志
    while sp.poll() is None:  # 检查子进程是否已经结束
        line = sp.stdout.readline()
        if line:  # 只有当有内容时才添加到输出列表
            if process_id in processes:
                processes[process_id]['output'].append(line)
                print(f"{process_id} : {line}")
    sp.wait()  # 等待子进程结束
    if process_id in processes:
        processes[process_id]['finished'] = True  # 设置'finished'标志为True


def execmdmanager(comand, shell=False):
    args = shlex.split(comand)
    sp = subprocess.Popen(args, stdout=subprocess.PIPE, encoding="utf8", shell=shell, stderr=subprocess.STDOUT)
    return sp


if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h':
            host = sys.argv[2]
        elif sys.argv[1] == '-p':
            port = int(sys.argv[2])
    app.run(debug=True, host=host, port=port)
