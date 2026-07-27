import os
import glob
import subprocess
import shutil

to_delete = glob.glob('scratch/*.py') + [
    'src/dashboard',
    'src/antigravity_inference_test.py',
    'src/check_partial_progress.py',
    'src/gemma_inference_test_ci_h1_v2.py',
    'src/gemma_inference_test_ci_ha_v1.py',
    'src/gemma_inference_test_ci_v1.py',
    'src/gemma_inference_test_ci_v2.py',
    'src/gemma_inference_test_ci_v4.py',
    'src/gemma_inference_test_ci_v5.py',
    'src/gemma_inference_test_refined.py',
    'src/gemma_inference_test_revised_zeroshot.py',
    'src/gemma_inference_test_sc_v1.py',
    'src/glm_inference_test_ci_v1.py',
    'src/grok_inference_test_ci_v1.py',
    'src/qwen3_32b_inference_test.py',
    'src/qwen_inference.py',
    'src/qwen_inference_test.py',
    'src/gemma_inference_full_v1_arabic_target.py',
    'src/gemma_inference_zeroshot_arabic_target_temp01.py',
    'src/gemma_inference_test.py',
    'src/gemma_inference_train.py',
    'src/gemma_inference.py',
    'src/arabert_finetune.py'
]

for path in to_delete:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        subprocess.run(['git', 'rm', '-r', '--ignore-unmatch', path])

subprocess.run(['git', 'add', '-A'])
subprocess.run(['git', 'commit', '-m', 'Clean up unused scripts and scratch folder'])
print('Cleanup complete.')
