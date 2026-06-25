function sys = get_system_info()


[res,gpu_info] = system("nvidia-smi -q");
[res,cpu_info]=system("lscpu");
[res,mem_info]=system("free -g -h -t");
% s = textscan(gpu_info,'%s','delimiter','\r')

sys.gpu_info = gpu_info;
sys.cpu_info = cpu_info;
sys.mem_info = mem_info;





