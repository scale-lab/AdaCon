import subprocess
import re
import time
import torch 
from thop import profile
from thop import clever_format

class Profiler:
    def __init__(self, platform='server'):
        self.pid = None
        self.platform = platform

    def start(self, use_cuda=True):
        if not use_cuda:
            print("ERROR: profiling on cpu is currently unsupported")
            return

        if not torch.cuda.is_available():
            print("ERROR: use_cuda selected for profiling while cuda is not available")
            return
        
        if self.platform == 'server':
            command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv", "-l", "1"]
        elif self.platform == 'nano':
            command = ["tegrastats", "--interval", "1000"]
        else:
            print("ERROR: unknown platform.")
            return

        self.pid = subprocess.Popen(command, stdout=subprocess.PIPE)

    def end(self):
        if not self.pid:
            print("ERROR: No process initialized, use Profiler.start_profiling to initialize a process")
        self.pid.kill()
        if self.platform == 'server':
            return self._parse_server_output(self.pid.stdout.readlines())
        elif self.platform == 'nano':
            return self._parse_nano_output(self.pid.stdout.readlines())
        else:
            print("ERROR: unknown platform.")
            return None, None, None

    def _parse_server_output(self, lines):
        '''
        Returns the peak memory in MiB
        '''
        lines = list(map(lambda l: int(l.decode("utf-8").strip().split(' ')[0]), lines[1:]))
        return max(lines), sum(lines)/len(lines), len(lines)
    
    def _parse_nano_output(self, lines):
        '''
        Returns the peak memory in MiB
        '''
        lines = list(map(lambda l: l.decode("utf-8").strip(), lines))
        gpu_power = []
        cpu_power = []
        total_power = []
        for line in lines:
            match = re.search('POM_5V_GPU (?P<gpu_power>[0-9]+)/[0-9]' ,line)
            power = float(match.group('gpu_power'))
            gpu_power.append(power)
            match = re.search('POM_5V_CPU (?P<cpu_power>[0-9]+)/[0-9]' ,line)
            power = float(match.group('cpu_power'))
            cpu_power.append(power)
            total_power.append(cpu_power[-1]+gpu_power[-1])
        # Discard the first 50 readings - warmup
        gpu_power = gpu_power[50:]
        cpu_power = cpu_power[50:]
        total_power = total_power[50:]
        return sum(gpu_power)/len(gpu_power), sum(cpu_power)/len(cpu_power), sum(total_power)/len(total_power)

    def get_inst_nano_power(self, lines):
        lines = list(map(lambda l: l.decode("utf-8").strip(), lines))
        line = lines[-1]
        match = re.search('POM_5V_GPU (?P<gpu_power>[0-9]+)/[0-9]' ,line)
        power = float(match.group('gpu_power'))
        gpu_power = power
        match = re.search('POM_5V_CPU (?P<cpu_power>[0-9]+)/[0-9]' ,line)
        power = float(match.group('cpu_power'))
        cpu_power = power

        return cpu_power + gpu_power

    def profile_params(self, model, num_branches = 1):
        def count_parameters(model):
            total_params = 0
            for name, parameter in model.named_parameters():
                param = parameter.numel()
                total_params+=param
            return total_params
        backbone_params = count_parameters(model.backbone)/(1000*1000)
        total_params = count_parameters(model)/(1000*1000)
        heads_params = (total_params - backbone_params)/num_branches
        return total_params, backbone_params, heads_params

    def profile_macs(self, model, input, num_branches = 1):
        total_macs, _ = profile(model, inputs=(input, ), verbose=False)
        total_macs = total_macs/(1024*1024*1024)
        backbone_macs, _ = profile(model.backbone, inputs=(input, ), verbose=False)
        backbone_macs = backbone_macs/(1024*1024*1024)
        
        return total_macs, backbone_macs, total_macs - backbone_macs

if __name__ == '__main__':
    from torch import randn, randint
    from torch.nn.functional import nll_loss
    from torchvision.models import resnet18, resnet34
    from edgify.models import resnet18 as resnet18_sparse
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profiler = Profiler(platform='nano')
    profiler.start(use_cuda=True)
    time.sleep(5)
    mem_sys_peak, mem_sys_avg, _ = profiler.end()

    profiler.start(use_cuda=True)
    model = resnet18_sparse().to(device)
    inputs = randn(64, 3, 224, 224, device=device)
    labels = randint(2, (64, ), device=device)
    loss_fn = nll_loss
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()

    mem_peak, mem_avg, prtime = profiler.end()
    mem_peak -= mem_sys_peak
    mem_avg -= mem_sys_avg
    
    print(f'Peak memory: {mem_peak:.4f} MB, Average memory: {mem_avg:.4f} MB, Time: {prtime} sec.')