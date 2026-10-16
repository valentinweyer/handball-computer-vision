"""Serial SDPA backend equality/time gate at SAM2's measured attention shapes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    rows = []
    with torch.inference_mode():
        for m, n, d in ((4096,4096,256),(4096,28736,256),(4096,12304,256)):
            q = torch.randn(1,1,m,d,device='cuda',dtype=torch.bfloat16)
            k = torch.randn(1,1,n,d,device='cuda',dtype=torch.bfloat16)
            v = torch.randn_like(k)
            values = {}
            row = {'qkd':[m,n,d], 'backends':{}}
            for name, backend in (('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION)):
                try:
                    with sdpa_kernel(backend):
                        for _ in range(3):
                            out = F.scaled_dot_product_attention(q,k,v)
                        values[name] = out.clone()
                        start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(20):
                            out = F.scaled_dot_product_attention(q,k,v)
                        end.record()
                        end.synchronize()
                        row['backends'][name] = {'ms':start.elapsed_time(end)/20}
                except RuntimeError as error:
                    row['backends'][name] = {'error':str(error)}
            if len(values) == 2:
                a,b = values['flash'],values['cudnn']
                mismatch = a != b
                indices = mismatch.flatten().nonzero()
                row.update(identical=torch.equal(a,b),different=int(mismatch.sum()),
                           first_differing_index=int(indices[0]) if indices.numel() else None,
                           max_abs=float((a.float()-b.float()).abs().max()),
                           speedup=row['backends']['flash']['ms']/row['backends']['cudnn']['ms'])
            rows.append(row)
            print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'torch':torch.__version__,'rows':rows},indent=2)+'\n')

if __name__ == '__main__':
    main()
