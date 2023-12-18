* Directory structure
- mha_weight: exported multi-head attention layer weight to be written into HLS files
- ret_weight: exported retentive layer weight to be written into HLS files
- ecelinux: HLS files
- python: retentive model
  - play_example: some playing code
  - torchscale: code from Microsoft torchscale/retnet (https://github.com/microsoft/torchscale)
- transformer: code from github (https://github.com/hyunwoongko/transformer)
- mha_compare.py: evaluate multi-head attention, show elapsed time and memory usage
- ret_compare.py: evaluate retentive layer, show elapsed time and memory usage
- mha_verify.py: verify hardware design, export weights
- ret_verify.py: verify hardware design, export weights
- run.sh: run mha_compare & ret_compare several times with different sequence length, use draw.py afterwards to generate the graph
- draw.py: generate graph for latency and memory usage between retentive and mha