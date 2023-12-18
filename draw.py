import matplotlib.pyplot as plt
import numpy as np

def read_file():
  MHA_FILE = ['MHA_result1.txt', 'MHA_result2.txt', 'MHA_result3.txt', 'MHA_result4.txt']
  RET_FILE = ['RET_result1.txt', 'RET_result2.txt', 'RET_result3.txt', 'RET_result4.txt']

  mha_exe_time = list()
  mha_memory = list()
  ret_exe_time = list()
  ret_memory = list()

  for i in range(0, len(MHA_FILE)):
    with open(MHA_FILE[i], 'r') as f:
      lines = f.readlines()
      data = np.array(lines).astype(np.float32)
      mha_exe_time.append(data[::2])
      mha_memory.append(data[1::2])

    with open(RET_FILE[i], 'r') as f:
      lines = f.readlines()
      data = np.array(lines).astype(np.float32)
      ret_exe_time.append(data[::2])
      ret_memory.append(data[1::2])
  mha_exe_time = np.array(mha_exe_time)
  mha_memory = np.array(mha_memory)
  ret_exe_time = np.array(ret_exe_time)
  ret_memory = np.array(ret_memory)
  mha_exe_time = mha_exe_time.mean(axis=0)
  mha_memory = mha_memory.mean(axis=0)
  ret_exe_time = ret_exe_time.mean(axis=0)
  ret_memory = ret_memory.mean(axis=0)
  return mha_exe_time, mha_memory, ret_exe_time, ret_memory


def draw():
  # Create some mock data

  
  num_of_tokens = np.array([1024, 2048, 3096, 4192, 5120, 6144, 7168])
  mha_exe_time, mha_memory, ret_exe_time, ret_memory = read_file()
  mha_memory  = mha_memory/100000000 # GB
  ret_memory  = ret_memory/100000000 # GB
  mha_color = 'tab:red'
  ret_color = 'tab:blue'
  label_color = 'k'
  line_width = 1.5

  # width = 0.1*num_of_tokens
  # width = 100

  fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)

  # ax1.set_xscale('log')
  axs[0].set_title('Memory usage & execution time for \n single layer auto-regressive inference')

  # axs[0].set_yscale('log')
  axs[0].set_ylabel('Memory Usage (GB)', color=label_color)  # we already handled the x-label with ax1
  axs[0].plot(num_of_tokens, mha_memory, color=mha_color, linewidth=line_width, label='Self-attention')
  axs[0].plot(num_of_tokens, ret_memory, color=ret_color, linewidth=line_width, label='Retentive')
  axs[0].tick_params(axis='y', labelcolor=label_color)
  axs[0].legend()

  axs[1].set_xlabel('# of tokens processed', color=label_color)
  axs[1].set_ylabel('Execution Time (s)', color=label_color)
  axs[1].plot(num_of_tokens, mha_exe_time, color=mha_color, linewidth=line_width, label='Self-attention')
  axs[1].plot(num_of_tokens, ret_exe_time, color=ret_color, linewidth=line_width, label='Retentive')
  axs[1].tick_params(axis='y', labelcolor=label_color)

  fig.tight_layout()  # otherwise t
  plt.savefig('fig.png', dpi=500)

if __name__ == '__main__':
  draw()
  