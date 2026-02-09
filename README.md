<h1 align="center"> Code Safety </h1>
<p align="center"> <b>...</b> 
</p>
<p align="center">

</p>  
<p align="center"> 

</p>


---

## 🛠️ Full Setup

### 1. Environment Setup

```bash
git clone https://github.com/.../....git && cd safety_code
conda create -n verl_safe python=3.10 -y
conda activate verl_safe
bash install.sh
```

### 2. Training

Run training for your target dataset and model size:

```bash
# For SecCodePLT+  
bash run/seccodeplt/hybrid-[0.5b|3b|7b].sh

# For APPS+  
bash run/apps-code/hybrid-[0.5b|3b|7b].sh

# For SafeSQL  
bash run/safesql/hybrid-[0.5b|3b|7b].sh
```

> 💾 Checkpoints are automatically saved to: `checkpoints/[proj_name]/[exp_name]`

### 3. Evaluation

Execute the evaluation pipeline after training:

```bash
# For SecCodePLT+
bash experiments/seccodeplt/seccodeplt_eval_main.sh

# For APPS+
bash experiments/safesql_and_apps/apps_eval_main.sh [path_to_exp_dir] [sharded|single]

# For SafeSQL
bash experiments/safesql_and_apps/safesql_eval_main.sh [path_to_exp_dir] [sharded|single]
```

> This codebase is inspired by [this paper](https://arxiv.org/abs/2505.22704). Thanks to the authors for their great work.
---

